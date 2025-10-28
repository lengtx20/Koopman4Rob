from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    SampleStamped,
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
    DataRearrangeConfig,
    RearrangeType,
)
from mcap_data_loader.utils.extra_itertools import Reusablizer, epairwise, take_skip
from mcap_data_loader.pipelines import NestedZip, NestedZipConfig, Merge, MergeConfig
from typing import Tuple, List
from config import Config
from functools import partial
from pprint import pprint
import torch


def create_mcap_dataloader(config: Config, datasets: list):
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    source_nodes = {}
    weights = {}

    nested = NestedZip[McapFlatBuffersEpisodeDataset](NestedZipConfig(depth=1))(
        datasets
    )
    for episodes in nested:
        key = episodes[0].config.data_root.name
        merged = Merge(MergeConfig(method="ChainMap"))(episodes)
        source_nodes[key] = nodes.IterableWrapper(
            Reusablizer(partial(epairwise, gap=config.pair_gap, fill_with_last=True))(
                merged
            )
        )
        weights[key] = 1.0

    node = nodes.MultiNodeWeightedSampler(
        source_nodes,
        weights,
        nodes.StopCriteria.ALL_DATASETS_EXHAUSTED,
    )

    def process_batched_sample(
        batched_samples: List[Tuple[SampleStamped, SampleStamped]],
    ) -> torch.Tensor:
        batched_list = []
        for sample in batched_samples:
            tensor_samples = []
            # convert to tensor and move to device
            for s in sample:
                tensor_sample = {}
                for key, value in s.items():
                    tensor_sample[key] = torch.from_numpy(value["data"]).to(
                        device=device, dtype=dtype
                    )
                tensor_samples.append(tensor_sample)
            sample_array = torch.concatenate(
                [
                    # state dim + action dim
                    tensor_samples[0][key]
                    for key in config.robot_action_keys + config.img_features_keys
                ]
                + [  # next state dim
                    tensor_samples[1][key] for key in config.robot_action_keys
                ]
            ).to(dtype=torch.float32)
            batched_list.append(sample_array)
        return torch.stack(batched_list)

    node = nodes.Batcher(node, config.batch_size, False)
    node = nodes.ParallelMapper(node, process_batched_sample, config.num_workers)
    node = nodes.Prefetcher(node, 10, 1000)
    # node = nodes.PinMemory(node, "cuda", 1000)
    # node = nodes.Unbatcher(node)
    return nodes.Loader(node, True)


def create_train_val_dataloader(config: Config):
    com_cfg = {
        "strict": False,
        "rearrange": DataRearrangeConfig(dataset=RearrangeType.SORT_STEM_DIGITAL),
    }
    zipping_datasets = (
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root=config.data_dir, keys=config.robot_action_keys, **com_cfg
            )
        ),
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root=f"{config.data_dir}_blip2_features",
                keys=config.img_features_keys,
                **com_cfg,
            )
        ),
    )
    splited_datasets = {"train": [], "val": []}
    for dataset in zipping_datasets:
        dataset.load()
        sample_datasets = list(dataset.read_stream())
        num = len(sample_datasets)
        split = config.train.train_val_split
        if isinstance(split, float):
            train_num = int(num * split)
            train = sample_datasets[:train_num]
            val = sample_datasets[train_num:]
        else:
            train, val = take_skip(sample_datasets, split[0], split[1])
            RearrangeType.rearrange(train, RearrangeType.SORT_STEM_DIGITAL)
            RearrangeType.rearrange(val, RearrangeType.SORT_STEM_DIGITAL)
        splited_datasets["train"].append(train)
        splited_datasets["val"].append(val)
    pprint(splited_datasets)
    # check the names and lengths are matching
    for stage, datasets in splited_datasets.items():
        if not all(len(v) == len(datasets[0]) for v in datasets):
            raise ValueError(
                f"Dataset lengths do not match in {stage}: {[len(v) for v in datasets]}"
            )
        for sample_sets in zip(*datasets):
            sample_sets: Tuple[McapFlatBuffersSampleDataset, ...]
            name = sample_sets[0].config.data_root.name
            for sample_set in sample_sets:
                if sample_set.config.data_root.name != name:
                    raise ValueError(
                        f"Dataset names do not match: {sample_set.config.data_root.name} vs {name}"
                    )
    return [
        create_mcap_dataloader(config, splited_datasets[key])
        for key in ["train", "val"]
    ]
