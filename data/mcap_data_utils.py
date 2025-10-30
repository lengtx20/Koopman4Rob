from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    SampleStamped,
    McapFlatBuffersEpisodeDataset,
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
    dl_cfg = config.data_loader
    dtype = getattr(torch, config.dtype)
    # zipping the internal episodes of the datasets
    nested = NestedZip[McapFlatBuffersEpisodeDataset](NestedZipConfig(depth=1))(
        datasets
    )
    source_nodes = {}
    weights = {}
    for index, episodes in enumerate(nested):
        # merge the zipped episodes
        merged = Merge(MergeConfig(method="ChainMap"))(episodes)
        source_node = nodes.IterableWrapper(
            Reusablizer(partial(epairwise, gap=dl_cfg.pair_gap, fill_with_last=True))(
                merged
            )
        )
        source_nodes[index] = source_node
        weights[index] = dl_cfg.weights[index] if dl_cfg.weights else 1.0

    if len(source_nodes) == 1:
        node = source_node
    else:
        node = nodes.MultiNodeWeightedSampler(
            source_nodes,
            weights,
            nodes.StopCriteria.ALL_DATASETS_EXHAUSTED,
        )

    # get the state and action dims
    first_sample = next(iter(nodes.Loader(node, False)))
    state_dim = sum(first_sample[0][key]["data"].shape[0] for key in dl_cfg.states)
    action_dim = sum(first_sample[0][key]["data"].shape[0] for key in dl_cfg.actions)
    print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")
    if config.model.state_dim == 0:
        config.model.state_dim = state_dim
    if config.model.action_dim == 0:
        config.model.action_dim = action_dim

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
                    tensor_sample[key] = torch.from_numpy(value["data"])
                tensor_samples.append(tensor_sample)
            sample_array = torch.concatenate(
                [
                    # state dim + action dim
                    tensor_samples[0][key]
                    for key in config.data_loader.states + config.data_loader.actions
                ]
                + [  # next state dim
                    tensor_samples[1][key] for key in config.data_loader.states
                ]
            ).to(dtype=dtype, device=torch.device(config.device))
            batched_list.append(sample_array)
        return torch.stack(batched_list)

    node = nodes.Batcher(node, dl_cfg.batch_size, False)
    node = nodes.ParallelMapper(node, process_batched_sample, dl_cfg.num_workers)
    if dl_cfg.prefetch_factor > 0:
        node = nodes.Prefetcher(
            node, dl_cfg.prefetch_factor, dl_cfg.prefetch_snapshot_frequency
        )
    if dl_cfg.pin_memory_device is not None:
        node = nodes.PinMemory(
            node, dl_cfg.pin_memory_device, dl_cfg.pin_memory_snapshot_frequency
        )
    loader = nodes.Loader(node, dl_cfg.restart_on_stop_iteration)
    return loader


def create_train_val_dataloader(config: Config):
    splited_datasets = {"train": [], "val": []}
    for dataset in config.datasets:
        sample_datasets = list(dataset)
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
