from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    SampleType,
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from mcap_data_loader.utils.extra_itertools import Reusablizer
from mcap_data_loader.piplines import NestedZip, NestedZipConfig, Merge, MergeConfig
from more_itertools import pairwise
from typing import Tuple, List

# from data.blip2_feature_extractor import Blip2ImageFeatureExtractor
import torch


def create_mcap_dataloader(
    model_path: str, datasets: list, batch_size: int, num_workers: int = 0, device=None
):
    # extractor = Blip2ImageFeatureExtractor(model_path, device)
    # extractor.load_model()
    # device=extractor.device; dtype=extractor.dtype
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
        source_nodes[key] = nodes.IterableWrapper(Reusablizer(pairwise)(merged))
        weights[key] = 1.0

    node = nodes.MultiNodeWeightedSampler(
        source_nodes,
        weights,
        nodes.StopCriteria.ALL_DATASETS_EXHAUSTED,
    )
    # prompt = "Open the cabinet door with the vertical black handle"
    arm_key = "/follow/arm/joint_state/position"
    eef_key = "/follow/eef/joint_state/position"
    cam_key = "/env_camera/color/image_raw"
    img_features_key = f"{cam_key}/features"

    def process_batched_sample(
        batched_samples: List[Tuple[SampleType, SampleType]],
    ) -> torch.Tensor:
        batched_list = []
        # mock_features = torch.zeros(256, dtype=extractor.dtype, device=extractor.device)
        for sample in batched_samples:
            tensor_samples = []
            for s in sample:
                tensor_sample = {}
                for key, value in s.items():
                    # if not value.flags.writeable:
                    #     value = value.copy()
                    tensor_sample[key] = torch.from_numpy(value).to(
                        device=device, dtype=dtype
                    )
                tensor_samples.append(tensor_sample)
            sample_array = torch.concatenate(
                [
                    # state dim
                    tensor_samples[0][arm_key],
                    tensor_samples[0][eef_key],
                    # action dim
                    # extractor.process_image(tensor_samples[0][cam_key], prompt).squeeze(
                    #     0
                    # ),
                    tensor_samples[0][img_features_key],
                    # mock_features,
                    # next state dim
                    tensor_samples[1][arm_key],
                    tensor_samples[1][eef_key],
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
    model_path: str, data_root, batch_size, num_workers, ratio: float = 0.8, device=None
):
    keys = [
        "/follow/arm/joint_state/position",
        "/follow/eef/joint_state/position",
        # "/env_camera/color/image_raw",
        "/env_camera/color/image_raw/features",
    ]
    datasets = (
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root=data_root, keys=keys[:2], strict=False
            )
        ),
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(
                data_root=f"{data_root}_blip2_features", keys=keys[2:], strict=False
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
        model_path, splited_datasets["train"], batch_size, num_workers, device
    )
    val_loader = create_mcap_dataloader(
        model_path, splited_datasets["val"], batch_size, num_workers, device
    )
    return train_loader, val_loader
