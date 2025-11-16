from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
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
    Map,
    MapConfig,
    PairWise,
    Slice,
    SliceConfig,
)
from mcap_data_loader.callers.stack import BatchStacker, BatchStackerConfig, DictBatch
from mcap_data_loader.callers import DictTuple
from typing import Tuple, List, Dict, Union
from config import Config
from pprint import pprint


def create_dataloader(
    config: Config, datasets: list
) -> Union[nodes.Loader, List[nodes.Loader]]:
    dl_cfg = config.data_loader
    stage = config.stage
    batch_stacker = BatchStacker(
        BatchStackerConfig(
            dtype=config.dtype,
            device=config.device,
            stack=dl_cfg.stack,
            backend_out="torch",
        )
    )
    batch_stacker.configure()
    # zipping the internal episodes of the datasets
    nested = NestedZip[McapFlatBuffersEpisodeDataset](NestedZipConfig(depth=1))(
        datasets
    )
    if dl_cfg.horizon is not None:
        tupler_cfg = dl_cfg.horizon
        tupler_cls = Horizon
        dict_tuple_depth = 2
    elif dl_cfg.pairwise is not None:
        tupler_cfg = dl_cfg.pairwise
        tupler_cls = PairWise
        dict_tuple_depth = 1
    else:
        raise NotImplementedError("No tupler configuration provided.")
    step = dl_cfg.future_span
    source_nodes = {}
    weights = {}
    for index, zipped_episodes in enumerate(nested):
        # merge the zipped episodes
        merged = Merge(MergeConfig(replace=True))(zipped_episodes)
        tupled = tupler_cls(tupler_cfg)(merged)
        if config.stage == "infer":
            final_pipe = Slice(SliceConfig(step=step))(tupled)
        else:
            final_pipe = tupled
        dict_tupled = Map(
            MapConfig(
                callable=DictTuple(
                    dl_cfg.dict_tuple.model_copy(update={"depth": dict_tuple_depth})
                )
            )
        )(final_pipe)
        source_node = nodes.IterableWrapper(dict_tupled)
        source_nodes[index] = source_node
        weights[index] = dl_cfg.weights[index] if dl_cfg.weights else 1.0
    if stage == "infer" or len(source_nodes) == 1:
        indexed_nodes = source_nodes
    else:
        indexed_nodes = {
            0: nodes.MultiNodeWeightedSampler(
                source_nodes,
                weights,
                nodes.StopCriteria.ALL_DATASETS_EXHAUSTED,
                seed=config.seed,
            )
        }
    all_loaders = []
    batch_size = dl_cfg.batch_size
    if stage == "infer":
        batch_size = 1
    for index, base_node in indexed_nodes.items():
        node = nodes.Batcher(base_node, batch_size, dl_cfg.drop_last)
        node = nodes.ParallelMapper(node, batch_stacker, **dl_cfg.parallel.model_dump())
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
    return all_loaders[0] if stage != "infer" else all_loaders


def create_dataloaders(config: Config) -> Dict[str, nodes.Loader[DictBatch]]:
    print("Creating data loaders...")
    if config.stage == "train":
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
        splited_datasets = {config.stage: list(config.datasets)}
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
