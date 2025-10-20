from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    SampleType,
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from mcap_data_loader.utils.extra_itertools import Reusablizer, epairwise
from mcap_data_loader.piplines import NestedZip, NestedZipConfig, Merge, MergeConfig
from typing import Tuple, List
from config import Config
from functools import partial
import torch


def create_mcap_dataloader(
    config: Config, datasets: list, batch_size: int, num_workers: int = 0, device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        batched_samples: List[Tuple[SampleType, SampleType]],
    ) -> torch.Tensor:
        batched_list = []
        for sample in batched_samples:
            tensor_samples = []
            # convert to tensor and move to device
            for s in sample:
                tensor_sample = {}
                for key, value in s.items():
                    tensor_sample[key] = torch.from_numpy(value).to(
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

    node = nodes.Batcher(node, batch_size, False)
    node = nodes.ParallelMapper(node, process_batched_sample, num_workers)
    node = nodes.Prefetcher(node, 10, 1000)
    # node = nodes.PinMemory(node, "cuda", 1000)
    # node = nodes.Unbatcher(node)
    return nodes.Loader(node, True)


def create_train_val_dataloader(
    config: Config,
    batch_size,
    num_workers,
    ratio: float = 0.8,
    device=None,
):
    datasets = (
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root=config.data_dir, keys=config.robot_action_keys, strict=False
            )
        ),
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root=f"{config.data_dir}_blip2_features",
                keys=config.img_features_keys,
                strict=False,
            )
        ),
    )
    splited_datasets = {"train": [], "val": []}
    for dataset in datasets:
        dataset.load()
        sample_datasets = list(dataset.read_stream())
        num = len(sample_datasets)
        train_num = int(num * ratio)
        splited_datasets["train"].append(sample_datasets[:train_num])
        splited_datasets["val"].append(sample_datasets[train_num:])
    # check the names and lengths are matching
    for key, value in splited_datasets.items():
        if not all(len(v) == len(value[0]) for v in value):
            raise ValueError(
                f"Dataset lengths do not match in {key}: {[len(v) for v in value]}"
            )
        for sample_sets in zip(*value):
            sample_sets: Tuple[McapFlatBuffersSampleDataset, ...]
            name = sample_sets[0].config.data_root.name
            for sample_set in sample_sets:
                if sample_set.config.data_root.name != name:
                    raise ValueError(
                        f"Dataset names do not match: {sample_set.config.data_root.name} vs {name}"
                    )
    train_loader = create_mcap_dataloader(
        config, splited_datasets["train"], batch_size, num_workers, device
    )
    val_loader = create_mcap_dataloader(
        config, splited_datasets["val"], batch_size, num_workers, device
    )
    return train_loader, val_loader
