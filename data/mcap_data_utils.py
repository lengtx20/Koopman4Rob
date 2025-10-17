from torchdata import nodes
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    SampleType,
)
from mcap_data_loader.utils.extra_itertools import Reusablizer
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from more_itertools import pairwise
from typing import Tuple, List
from data.blip2_feature_extractor import Blip2ImageFeatureExtractor
import torch


def create_mcap_dataloader(
    model_path: str, datasets: list, batch_size: int, num_workers: int = 0, device=None
):
    extractor = Blip2ImageFeatureExtractor(model_path, device)
    extractor.load_model()

    source_nodes = {}
    weights = {}
    for episode in datasets:
        episode: McapFlatBuffersSampleDataset
        source_nodes[episode.config.data_root] = nodes.IterableWrapper(
            Reusablizer(pairwise)(episode)
        )
        weights[episode.config.data_root] = 1.0

    node = nodes.MultiNodeWeightedSampler(
        source_nodes,
        weights,
        nodes.StopCriteria.ALL_DATASETS_EXHAUSTED,
    )
    prompt = "Open the cabinet door with the vertical black handle"
    arm_key = "/follow/arm/joint_state/position"
    eef_key = "/follow/eef/joint_state/position"
    cam_key = "/env_camera/color/image_raw"

    def process_batched_sample(
        batched_samples: List[Tuple[SampleType, SampleType]],
    ) -> torch.Tensor:
        batched_list = []
        mock_features = torch.zeros(256, dtype=extractor.dtype, device=extractor.device)
        for sample in batched_samples:
            tensor_samples = []
            for s in sample:
                tensor_sample = {}
                for key, value in s.items():
                    if not value.flags.writeable:
                        value = value.copy()
                    tensor_sample[key] = torch.from_numpy(value).to(
                        device=extractor.device, dtype=extractor.dtype
                    )
                tensor_samples.append(tensor_sample)
            sample_array = torch.concatenate(
                [
                    # state dim
                    tensor_samples[0][arm_key],
                    tensor_samples[0][eef_key],
                    # action dim
                    extractor.process_image(tensor_samples[0][cam_key], prompt).squeeze(
                        0
                    ),
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
    model_path: str, data_root, batch_size, num_workers, device=None
):
    keys = [
        "/follow/arm/joint_state/position",
        "/follow/eef/joint_state/position",
        "/env_camera/color/image_raw",
    ]
    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root=data_root, keys=keys, strict=False
        )
    )
    dataset.load()
    sample_datasets = list(dataset.read_stream())
    num = len(sample_datasets)
    train_num = int(num * 0.8)
    train_loader = create_mcap_dataloader(
        model_path, sample_datasets[:train_num], batch_size, num_workers, device
    )
    val_loader = create_mcap_dataloader(
        model_path, sample_datasets[train_num:], batch_size, num_workers, device
    )
    return train_loader, val_loader
