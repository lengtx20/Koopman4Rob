from mcap_data_loader.datasets.mcap_dataset import (
    DataRearrangeConfig,
    RearrangeType,
    get_config_and_class_type,
    to_episodic_sequence,
)
from config import DataLoaderConfig
from pathlib import Path
import torch


all_keys = {}


def _concat_values(values):
    result = []
    for v in values:
        result.extend(v)
    return result


def get_datasets(data_name):
    print(f"Loading datasets for {data_name}...")
    global all_keys

    base_root = Path(f"mcap_data/{data_name}")
    root_dir = base_root if base_root.is_dir() else base_root.parent
    feature_root_dir = Path(f"{root_dir}_blip2_features")
    feature_root = (
        feature_root_dir if base_root.is_dir() else feature_root_dir / base_root.name
    )
    feature_suffix = "features_proj"
    image_keys = ["/env_camera/color/image_raw"]
    all_keys = {
        "state": {base_root: ["/follow/arm/joint_state/position"]},
        "action": {feature_root: [f"{cam}/{feature_suffix}" for cam in image_keys]},
    }
    common = {
        "strict": False,
        "rearrange": DataRearrangeConfig(dataset=RearrangeType.SORT_STEM_DIGITAL),
    }
    print(f"{base_root=}, {feature_root=}")
    datasets = []
    for key_dict in all_keys.values():
        for data_root in tuple(key_dict.keys()):
            if Path(data_root).exists():
                config_cls, dataset_cls = get_config_and_class_type(data_root)
                datasets.append(
                    to_episodic_sequence(
                        dataset_cls(
                            config_cls(
                                data_root=data_root, keys=key_dict[data_root], **common
                            )
                        )
                    )
                )
            else:
                print(
                    f"Warning: data_root {data_root} with keys {key_dict.pop(data_root)} does not exist, skipping."
                )

    print(f"Datasets used: {datasets}")
    return datasets


def get_data_loader_config(**kwargs) -> DataLoaderConfig:
    stack = {}
    cur_stack_no_index = {}
    for i, prefix in enumerate(["cur_", "next_"]):
        for input_key, value in all_keys.items():
            cat_key = f"{prefix}{input_key}"
            # skip action for next_
            if i > 0 and input_key == "action":
                continue
            else:
                raw_keys = _concat_values(value.values())
                if i == 0:
                    cur_stack_no_index[cat_key] = raw_keys
            stack_keys = [f"{i}{key}" for key in raw_keys]
            stack[cat_key] = stack_keys
    config = DataLoaderConfig(stack=stack, **kwargs)
    return config


class LossCalculator:
    def __init__(self, name: str = "MSELoss"):
        self._func = getattr(torch.nn, name)()

    def __call__(self, predictions: torch.Tensor, batch_data: dict) -> torch.Tensor:
        return self._func(predictions, batch_data["next_state"])
