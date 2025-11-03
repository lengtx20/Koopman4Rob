from airbot_data_collection.common.systems.basis import SystemMode
from airbot_data_collection.common.environments.grouped import (
    GroupedEnvironment,
    GroupedEnvironmentConfig,
    GroupsSendActionConfig,
)
from airbot_data_collection.common.systems.grouped import (
    SystemSensorComponentGroupsConfig,
    AutoControlConfig,
)

# from airbot_data_collection.common.devices.cameras.v4l2 import (
#     V4L2Camera as Camera,
#     V4L2CameraConfig as CameraConfig,
# )
from airbot_data_collection.common.devices.cameras.mock import (
    MockCamera as Camera,
    MockCameraConfig as CameraConfig,
)
# from airbot_ie.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfigI

from airbot_ie.robots.airbot_play_mock import AIRBOTPlay, AIRBOTPlayConfig
from pydantic import BaseModel
from config import Config
from collections import ChainMap
from data.mcap_data_utils import BatchProcessor, DictBatch
from data.blip2_feature_extractor import Blip2ImageFeatureExtractor
from interactor import InteractorBasis
from pathlib import Path
from pprint import pprint
from typing import Literal
import torch


class ExtractorConfig(BaseModel):
    """Configuration for the feature extractor."""

    model_path: Path
    """Path to the pre-trained BLIP2 model."""
    prompt: str
    """Prompt for the BLIP2 model."""
    enable: bool = True
    """Whether to enable the feature extractor."""

    def model_post_init(self, context):
        if not self.model_path.exists():
            raise FileNotFoundError(f"BLIP2 model not found at {self.model_path}")
        if not self.prompt:
            raise ValueError("Prompt for BLIP2 model cannot be empty.")


class InteractorConfig(BaseModel):
    """Configuration for the interactor between the model and the environment."""

    extractor: ExtractorConfig
    open_loop_predict: bool = False
    """Whether to perform open-loop prediction during inference."""
    """Configuration for the feature extractor."""
    action_from: Literal["model", "data_loader", "none"] = "model"
    """Whether to use actions from the data loader during inference.
    If False, actions will be taken from the model."""
    model_input_from: Literal["env", "data_loader"] = "env"
    # show_image: bool = False
    # """Whether to display images during inference."""
    # image_transform: bool = False
    # """Whether to apply image transformations during inference."""


class Interactor(InteractorBasis):
    def __init__(self, config: InteractorConfig):
        self.config = config

    def add_config(self, config: Config):
        stack_dl = config.data_loader.stack
        stack_env = {}
        for cat_key, keys in stack_dl.items():
            if "next" in cat_key:
                continue
            stack_env[cat_key] = []
            for key in keys:
                _, raw_key = key.split("/", 1)
                stack_env[cat_key].append(raw_key)
        self.use_extractor = self.config.extractor.enable
        if self.use_extractor:
            self.extractor = Blip2ImageFeatureExtractor(
                model_path=self.config.extractor.model_path
            )
            self.extractor.load_model()
            stack_based = (
                stack_dl if self.config.model_input_from == "data_loader" else stack_env
            )
            torch_cat_keys = ("cur_action",)
            torch_stack = {key: stack_based[key] for key in torch_cat_keys}
            for key in torch_cat_keys:
                stack_env.pop(key, None)
            pprint(f"DataLoader stack:\n{stack_dl}")
            pprint(f"Interactor torch_stack:\n{torch_stack}")
        pprint(f"Interactor plain_stack:\n{stack_env}")
        self._np_batcher = BatchProcessor(
            config.dtype, config.device, stack_env, "numpy"
        )
        self._torch_batcher = BatchProcessor(
            config.dtype, config.device, torch_stack, "torch"
        )
        self._torch_dtype = self._torch_batcher.torch_dtype
        self._device = self._torch_batcher.device
        image_keys = ["/env_camera/color/image_raw"]
        self.from_keys = (
            image_keys
            if self.config.model_input_from == "env"
            else [f"0{key}" for key in image_keys]
        )
        self.to_keys = [f"{key}/features_proj" for key in self.from_keys]
        print(f"Extractor from_keys: {self.from_keys}, to_keys: {self.to_keys}")
        self._shared_config = config

    def add_first_batch(self, batch: DictBatch):
        print(f"{list(batch.keys())=}")
        if batch:
            reset_action = batch["cur_state"][0].tolist()
        else:
            reset_action = None
        env = GroupedEnvironment(
            GroupedEnvironmentConfig(
                components=SystemSensorComponentGroupsConfig(
                    groups=["/"] * 2,
                    names=["follow", "env_camera"],
                    roles=["l", "o"],
                    instances=[
                        AIRBOTPlay(AIRBOTPlayConfig()),
                        Camera(CameraConfig(camera_index=None)),
                    ],
                ),
                reset_action=GroupsSendActionConfig(
                    groups=["/"] if reset_action else [],
                    action_values=[reset_action],
                    modes=[SystemMode.RESETTING],
                ),
                auto_control=AutoControlConfig(groups=[]),
            )
        )
        if not env.configure():
            raise RuntimeError("Failed to configure the interactor environment.")
        env.reset()
        self.env = env
        self._action = GroupsSendActionConfig(
            groups=["/"],
            action_values=[[]],
            modes=[SystemMode.SAMPLING],
        )
        # get the state and action dims
        state_dim = batch["cur_state"].shape[1]
        action_dim = batch["cur_action"].shape[1]
        print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")
        config = self._shared_config
        if config.model.state_dim == 0:
            config.model.state_dim = state_dim
        if config.model.action_dim == 0:
            config.model.action_dim = action_dim

    def _send_action(self, action: list):
        self._action.action_values[0] = action
        self.env.input(self._action)

    def _set_model_output(self, pred: torch.Tensor):
        return self._send_action(pred[0].tolist())

    def _set_batch(self, batch: DictBatch):
        return self._send_action(batch["cur_state"][0].tolist())

    def update(self, prediction: torch.Tensor, batch: DictBatch):
        if self.config.action_from == "data_loader":
            self._set_batch(batch)
        elif self.config.action_from == "model":
            self._set_model_output(prediction)

    def get_model_input(
        self, last_prediction: torch.Tensor, batch: DictBatch
    ) -> DictBatch:
        data = (
            batch
            if self.config.model_input_from == "data_loader"
            else self._np_batcher([self.env.output().observation])
        )
        if self.config.open_loop_predict and last_prediction is not None:
            data["cur_state"] = last_prediction
        if self.use_extractor:
            features = {}
            for from_key, to_key in zip(self.from_keys, self.to_keys):
                features[to_key] = {
                    "data": self.extractor.process_image(
                        data[from_key][0], self.config.extractor.prompt
                    )["features_proj"].squeeze(0)
                }
            # print(f"{features.keys()=}")
            batched_features = self._torch_batcher([features])
            return ChainMap(batched_features, data)
        return data

    def shutdown(self):
        return self.env.shutdown()


def get_interactor() -> Interactor:
    interactor_cfg = InteractorConfig(
        extractor=ExtractorConfig(
            model_path=Path("pretrained_models/blip2-itm-vit-g"),
            prompt="The end effector of the robotic arm tries to get close to the QR code attached to the cabinet.",
            enable=True,
        ),
        open_loop_predict=False,
        action_from="data_loader",
        model_input_from="data_loader",
    )
    interactor = Interactor(interactor_cfg)
    return interactor
