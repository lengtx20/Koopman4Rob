from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    SampleStamped,
    McapFlatBuffersEpisodeDataset,
    RearrangeType,
)
from mcap_data_loader.utils.extra_itertools import take_skip
from mcap_data_loader.pipelines import (
    NestedZip,
    NestedZipConfig,
    Merge,
    MergeConfig,
    PairWise,
    PairWiseConfig,
    DictTuple,
    DictTupleConfig,
)
from typing import Tuple, List, Dict, Union, Literal
from config import Config
from pprint import pprint
from collections import ChainMap
import torch
import numpy as np


ArrayType = Union[torch.Tensor, np.ndarray]
DictBatch = ChainMap[str, Union[ArrayType, List[ArrayType], int]]


class BatchProcessor:
    def __init__(
        self,
        dtype: str,
        device: str,
        stack: Dict[str, List[str]],
        backend: Literal["torch", "numpy"] = "numpy",
    ):
        self.torch_dtype = getattr(torch, dtype)
        self.np_dtype = getattr(np, dtype)
        self.device = torch.device(device)
        self.stack = stack
        if backend == "numpy":
            self.empty_func = lambda data: np.empty(data, dtype=self.np_dtype)
            self.concat_func = lambda arrays: np.concatenate(arrays, axis=-1)
            self.stack_func = np.hstack
            self.convert_func = self._np_to_torch
        else:
            self.empty_func = lambda data: torch.empty(
                data, dtype=self.torch_dtype, device=self.device
            )
            self.concat_func = lambda tensors: torch.cat(tensors, dim=-1)
            self.stack_func = torch.hstack
            self.convert_func = lambda x: x
        keys_to_stack = set()
        for keys in stack.values():
            keys_to_stack.update(keys)
        self.keys_to_stack = keys_to_stack

    def _np_to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=self.device, non_blocking=True)

    def __call__(self, batched_samples: List[SampleStamped]) -> DictBatch:
        batch_size = len(batched_samples)
        # allocate memory for each key
        batched: DictBatch = {}
        keys_no_stack = set()
        for key, value in batched_samples[0].items():
            if key in self.keys_to_stack:
                batched[key] = self.empty_func((batch_size, *value["data"].shape))
            else:
                batched[key] = []
                keys_no_stack.add(key)
        # fill in data
        for i, sample in enumerate(batched_samples):
            for key in self.keys_to_stack:
                batched[key][i] = sample[key]["data"]
            for key in keys_no_stack:
                batched[key].append(sample[key])
        # stack and move to device
        # TODO: use multi-treaded pin_memory and use a new cuda stream to copy asynchronously
        # TODO: test the performance vs tensor-dict
        final_batched = {}
        for catkey, keys in self.stack.items():
            final_batched[catkey] = self.convert_func(
                self.concat_func([batched.pop(key) for key in keys])
            )
        batched["batch_size"] = batch_size
        # keep the remaining batched dict unstacked
        return ChainMap(final_batched, batched)


def create_dataloader(
    config: Config, datasets: list
) -> Union[nodes.Loader, Dict[int, nodes.Loader]]:
    dl_cfg = config.data_loader
    batch_processor = BatchProcessor(config.dtype, config.device, dl_cfg.stack)
    # zipping the internal episodes of the datasets
    nested = NestedZip[McapFlatBuffersEpisodeDataset](NestedZipConfig(depth=1))(
        datasets
    )
    source_nodes = {}
    weights = {}
    for index, zipped_episodes in enumerate(nested):
        # merge the zipped episodes
        merged = Merge(MergeConfig(method="ChainMap"))(zipped_episodes)
        # TODO: use a more common past_future wrapper?
        paired = PairWise(PairWiseConfig(gap=dl_cfg.pair_gap, fill_with_last=True))(
            merged
        )
        dict_tupled = DictTuple(
            DictTupleConfig(
                depth=config.data_loader.dict_tuple_depth, separate_key=False
            )
        )(paired)
        source_node = nodes.IterableWrapper(dict_tupled)
        source_nodes[index] = source_node
        weights[index] = dl_cfg.weights[index] if dl_cfg.weights else 1.0
    if config.mode == "infer" or len(source_nodes) == 1:
        indexed_nodes = source_nodes
    else:
        indexed_nodes = {
            0: nodes.MultiNodeWeightedSampler(
                source_nodes, weights, nodes.StopCriteria.ALL_DATASETS_EXHAUSTED
            )
        }

    all_loaders = {}
    for index, base_node in indexed_nodes.items():
        if dl_cfg.prefetch_factor > 0:
            node = nodes.Prefetcher(
                base_node, dl_cfg.prefetch_factor, dl_cfg.prefetch_snapshot_frequency
            )
        node = nodes.Batcher(
            base_node,
            dl_cfg.batch_size if config.mode != "infer" else 1,
            dl_cfg.drop_last,
        )
        node = nodes.ParallelMapper(node, batch_processor, dl_cfg.num_workers)
        if dl_cfg.pin_memory_device is not None:
            raise NotImplementedError(
                "Pin memory is not implemented in this MCAP data loader now."
            )
            # NOTE: pin_memory should be applied before tensor being moved to device
            # TODO: pin memory by specified dict keys or auto skip np.ndarrays
            # node = nodes.PinMemory(
            #     node, dl_cfg.pin_memory_device, dl_cfg.pin_memory_snapshot_frequency
            # )
        loader = nodes.Loader(node, dl_cfg.restart_on_stop_iteration)
        all_loaders[index] = loader

    # get the state and action dims
    # first_sample = next(iter(nodes.Loader(indexed_nodes[0], False)))
    # state_dim = sum(first_sample[0][key]["data"].shape[0] for key in dl_cfg.stack)
    # action_dim = sum(first_sample[0][key]["data"].shape[0] for key in dl_cfg.actions_)
    # print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")
    # if config.model.state_dim == 0:
    #     config.model.state_dim = state_dim
    # if config.model.action_dim == 0:
    #     config.model.action_dim = action_dim

    return all_loaders[0] if config.mode != "infer" else all_loaders


def create_dataloaders(config: Config) -> Dict[str, nodes.Loader[DictBatch]]:
    print("Creating data loaders...")
    if config.mode == "train":
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
    else:
        splited_datasets = {config.mode: config.datasets}
    pprint(splited_datasets)
    print("checking the names and lengths are matching...")
    data_loaders = {}
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
        data_loaders[stage] = create_dataloader(config, datasets)
    print("Data loaders created.")
    return data_loaders
