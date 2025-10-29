from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
    DataRearrangeConfig,
    RearrangeType,
)
from config import DataLoaderConfig
from pathlib import Path


# parser = argparse.ArgumentParser(description="Data loading configuration")
# parser.add_argument

task_name = "reach3tags"
base_root = f"mcap_data/{task_name}"
feature_suffix = "features_proj"
image_keys = ["/env_camera/color/image_raw"]
all_keys = {
    "state": {base_root: ["/follow/arm/joint_state/position"]},
    "action": {
        f"{base_root}_blip2_features": [f"{cam}/{feature_suffix}" for cam in image_keys]
    },
}
common = {
    "strict": False,
    "rearrange": DataRearrangeConfig(dataset=RearrangeType.SORT_STEM_DIGITAL),
}

datasets = []
for key_type, key_dict in all_keys.items():
    for data_root in tuple(key_dict.keys()):
        if Path(data_root).exists():
            datasets.append(
                McapFlatBuffersEpisodeDataset(
                    McapFlatBuffersEpisodeDatasetConfig(
                        data_root=data_root,
                        keys=key_dict[data_root],
                        **common,
                    )
                )
            )
        else:
            print(
                f"Warning: data_root {data_root} with keys {key_dict.pop(data_root)} does not exist, skipping."
            )

print(f"Datasets used: {datasets}")


def _concat_values(values):
    result = []
    for v in values:
        result.extend(v)
    return result


def get_datasets():
    return datasets


def get_data_loader_config(**kwargs) -> DataLoaderConfig:
    config = DataLoaderConfig(
        states=_concat_values(all_keys["state"].values()),
        actions=_concat_values(all_keys["action"].values()),
        **kwargs,
    )
    print(f"{config.states=}, {config.actions=}")
    return config
