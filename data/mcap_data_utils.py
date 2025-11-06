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
    Horizon,
    DictTuple,
)
from typing import Tuple, List, Dict, Union, Literal
from config import Config
from pprint import pprint
from collections import ChainMap, defaultdict
from basis import DictBatch
from config import StackType, NormStackValue
from pydantic import BaseModel
from airbot_data_collection.common.utils.array_like import get_tensor_device_auto
import torch
import numpy as np


class BatchStackerConfig(BaseModel):
    dtype: str = ""
    """Data type for the stacked tensors/arrays."""
    device: str = ""
    """Device to move the stacked tensors to (only for torch backend)."""
    stack: StackType
    """Configuration for stacking keys."""
    backend_in: Literal["torch", "numpy"] = "numpy"
    """The input data backend."""
    backend_out: Literal["torch", "numpy", "list"] = "numpy"
    """The output data backend."""


class BatchProcessor:
    def __init__(self, config: BatchStackerConfig):
        self.torch_dtype = getattr(torch, config.dtype or "float32")
        self.np_dtype = getattr(np, config.dtype or "float32")
        self.torch_device = get_tensor_device_auto(config.device)
        if config.backend_in == "numpy":
            self.empty_func = lambda data: np.empty(data, dtype=self.np_dtype)
            self.concat_func = lambda arrays: np.concatenate(arrays, axis=-1)
            self.stack_func = np.hstack
        else:
            self.empty_func = lambda data: torch.empty(
                data, dtype=self.torch_dtype, device=self.torch_device
            )
            self.concat_func = lambda tensors: torch.cat(tensors, dim=-1)
            self.stack_func = torch.hstack
        # determine conversion function
        if config.backend_in == config.backend_out:
            self.convert_func = lambda x: x
        elif config.backend_out == "list":
            self.convert_func = self._to_list
        elif config.backend_out == "numpy":
            self.convert_func = self._torch_to_np
        else:
            self.convert_func = self._np_to_torch
        self.stack = self._normalize_stack_config(config.stack)
        keys_to_stack = {}
        v_keys = defaultdict(list)
        for cat_key, list_keys in self.stack.items():
            col_num = len(list_keys[0])
            for c in range(col_num):
                for r, keys in enumerate(list_keys):
                    v_keys[cat_key].append(keys[c])
                    keys_to_stack[keys[c]] = (cat_key, c, r)
        self.keys_to_stack = keys_to_stack
        self._first_call = True

    def _normalize_stack_config(self, stack: StackType) -> Dict[str, NormStackValue]:
        def process_value(config):
            if isinstance(config, tuple):
                keys, prefixes = config
                return [[f"{p}{k}" for k in keys] for p in prefixes]
            else:
                first = config[0]
                if isinstance(first, str):
                    return [config]
                else:
                    return config

        return {k: process_value(v) for k, v in stack.items()}

    def _np_to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(device=self.torch_device, non_blocking=True)

    def _torch_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    def _to_list(self, data: Union[np.ndarray, torch.Tensor]) -> list:
        return data.tolist()

    def _reset_buffers(self):
        for cat_key, c_shapes in self._batch_stack_shape.items():
            for c, shape in enumerate(c_shapes):
                self._batch_stack[cat_key][c] = self.empty_func(shape)
        for key in self._keys_no_stack:
            self._batch_list[key] = []

    def __call__(self, batched_samples: List[SampleStamped]) -> DictBatch:
        batch_size = len(batched_samples)
        if self._first_call:
            batch_stack_shape: Dict[str, List[tuple]] = {}
            batch_stack_empty: Dict[str, list] = {}
            first_sample = batched_samples[0]
            for cat_key, list_keys in self.stack.items():
                first_row = list_keys[0]
                row_num = len(list_keys)
                batch_stack_shape[cat_key] = []
                batch_stack_empty[cat_key] = []
                for c, key in enumerate(first_row):
                    batch_stack_shape[cat_key].append(
                        (batch_size, row_num, *first_sample[key]["data"].shape)
                    )
                    batch_stack_empty[cat_key].append(None)
            self._batch_stack_shape = batch_stack_shape
            self._batch_stack = batch_stack_empty
            self._keys_no_stack = first_sample.keys() - self.keys_to_stack.keys()
            self._batch_list: Dict[str, list] = {}
            self._first_call = False
        # allocate memory
        self._reset_buffers()
        # fill in data
        for i, sample in enumerate(batched_samples):
            for key, config in self.keys_to_stack.items():
                cat_key, c, r = config
                self._batch_stack[cat_key][c][i, r] = sample[key]["data"]
            for key in self._keys_no_stack:
                self._batch_list[key].append(sample[key]["data"])
        # stack and move to device
        # TODO: use multi-treaded pin_memory and use a new cuda stream to copy asynchronously
        # TODO: test the performance vs tensor-dict
        final_batched = {}
        for catkey, c_data in self._batch_stack.items():
            final_batched[catkey] = self.convert_func(self.concat_func(c_data))
        # keep the remaining batched dict unstacked
        return ChainMap(final_batched, self._batch_list, {"batch_size": batch_size})


def create_dataloader(
    config: Config, datasets: list
) -> Union[nodes.Loader, List[nodes.Loader]]:
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
        paired = Horizon(dl_cfg.horizon)(merged)
        dict_tupled = DictTuple(dl_cfg.dict_tuple)(paired)
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

    all_loaders = []
    if config.mode == "infer":
        print("Changing batch size to 1 for inference.")
        dl_cfg.batch_size = 1
    for index, base_node in indexed_nodes.items():
        if dl_cfg.prefetch_factor > 0:
            node = nodes.Prefetcher(
                base_node, dl_cfg.prefetch_factor, dl_cfg.prefetch_snapshot_frequency
            )
        node = nodes.Batcher(base_node, dl_cfg.batch_size, dl_cfg.drop_last)
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
        all_loaders.append(loader)
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
        splited_datasets = {config.mode: list(config.datasets)}
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
