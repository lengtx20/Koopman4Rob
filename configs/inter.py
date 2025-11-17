from airbot_data_collection.common.systems.basis import SystemMode
from airbot_data_collection.common.live_data.grouped import (
    GroupedSystemDataSourceConfig,
    GroupedSystemDataSource,
    GroupsSendActionConfig,
)
from airbot_data_collection.common.systems.grouped import (
    SystemSensorComponentGroupsConfig,
    AutoControlConfig,
)
from airbot_data_collection.common.devices.cameras.v4l2 import (
    V4L2Camera as Camera,
    V4L2CameraConfig as CameraConfig,
)
from airbot_ie.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig

# from airbot_data_collection.common.devices.cameras.mock import (
#     MockCamera as Camera,
#     MockCameraConfig as CameraConfig,
# )
# from airbot_ie.robots.airbot_play_mock import AIRBOTPlay, AIRBOTPlayConfig
from mcap_data_loader.utils.basic import remove_util
from mcap_data_loader.callers.dict_map import DictMap, DictMapConfig
from pydantic import BaseModel
from config import Config, ConfigDict
from collections import ChainMap
from data.mcap_data_utils import BatchStacker, BatchStackerConfig, DictBatch
from data.blip2_feature_extractor import Blip2ImageFeatureExtractor
from interactor import InteractorBasis, YieldKey, SendValue
from pathlib import Path
from pprint import pformat
from typing import Literal, List
from more_itertools import collapse
import torch
# import cv2


class ExtractorConfig(BaseModel):
    """Configuration for the feature extractor."""

    model_config = ConfigDict(extra="forbid")

    model_path: Path
    """Path to the pre-trained BLIP2 model."""
    prompt: str
    """Prompt for the BLIP2 model."""
    enable: bool = True
    """Whether to enable the feature extractor."""
    keys_from: List[str]
    """Keys to extract features from."""
    keys_to: List[str] = []
    """Keys to store extracted features."""

    def model_post_init(self, context):
        if not self.model_path.exists():
            raise FileNotFoundError(f"BLIP2 model not found at {self.model_path}")
        if not self.prompt:
            raise ValueError("Prompt for BLIP2 model cannot be empty.")


class InteractorConfig(BaseModel):
    """Configuration for the interactor between the model and the environment."""

    model_config = ConfigDict(extra="forbid")

    extractor: ExtractorConfig
    """Configuration for the feature extractor."""
    open_loop_predict: bool = False
    """Whether to perform open-loop prediction during inference."""
    action_from: Literal["model", "data_loader", "none"] = "model"
    """Source of actions to send to the environment. 'model' uses model predictions,
    'data_loader' uses actions from the data loader, and 'none' sends no actions"""
    model_input_from: Literal["env", "data_loader"] = "env"
    """Source of model inputs. 'env' uses data from the environment, 
    'data_loader' uses data from the data loader."""
    # show_image: bool = False
    # """Whether to display images during inference."""
    # image_transform: bool = False
    # """Whether to apply image transformations during inference."""


class Interactor(InteractorBasis):
    config: InteractorConfig

    def _remove_prefix(self, keys: List[str]) -> List[str]:
        return [remove_util(key, "/", True) for key in keys]

    def add_config(self, config: Config):
        stack_dl = config.data_loader.stack
        stack_env = {}
        for cat_key, keys in stack_dl.items():
            if "next" in cat_key:
                continue
            stack_env[cat_key] = [self._remove_prefix(row_keys) for row_keys in keys]
        extract_cfg = self.config.extractor
        self.use_extractor = extract_cfg.enable
        model_input_from = self.config.model_input_from
        print(f"{model_input_from=}")
        if self.use_extractor:
            self.extractor = Blip2ImageFeatureExtractor(
                model_path=extract_cfg.model_path
            )
            self.extractor.load_model()
            stack_based = stack_dl if model_input_from == "data_loader" else stack_env
            torch_cat_keys = ("cur_action",)
            torch_stack = {key: stack_based[key] for key in torch_cat_keys}
            for key in torch_cat_keys:
                stack_env.pop(key, None)
            # torch batcher for extracted data (torch tensors)
            self._extra_batcher = BatchStacker(
                BatchStackerConfig(
                    dtype=config.dtype,
                    device=config.device,
                    stack=torch_stack,
                )
            )
            keys_from = extract_cfg.keys_from
            keys_to = extract_cfg.keys_to
            if model_input_from == "env":
                keys_from = self._remove_prefix(keys_from)
                keys_to = self._remove_prefix(keys_to)
            self.from_keys = keys_from
            self.to_keys = (
                [f"{key}/features_proj" for key in self.from_keys]
                if not keys_to
                else keys_to
            )
            print(f"Extract from_keys: {self.from_keys}, to_keys: {self.to_keys}")
            print(f"Interactor torch_stack:\n{pformat(torch_stack)}")
        print(f"DataLoader stack:\n{pformat(stack_dl)}")
        print(f"Environment stack:\n{pformat(stack_env)}")
        # np batcher for env data (numpy arrays)
        self._env_batcher = BatchStacker(
            BatchStackerConfig(
                dtype=config.dtype,
                device=config.device,
                stack=stack_env,
                backend_out="torch",
            )
        )
        self._shared_config = config
        self._env_keys = set(collapse(stack_env.values()))
        self._dic_map = DictMap(
            DictMapConfig(
                dtype=config.dtype,
                device=config.device,
                # TODO: torch will be much faster
                backend_out="torch",
                keys_include=self._env_keys,
                replace=True,
            )
        )
        self._init_live_data()

    def _init_live_data(self):
        # TODO: make the live data configuration more flexible
        live_data = GroupedSystemDataSource(
            GroupedSystemDataSourceConfig(
                components=SystemSensorComponentGroupsConfig(
                    groups=["/"] * 2,
                    names=["follow", "env_camera"],
                    roles=["l", "o"],
                    instances=[
                        AIRBOTPlay(AIRBOTPlayConfig()),
                        Camera(CameraConfig(camera_index="usb-0000:00:14.0-11")),
                    ],
                ),
                auto_control=AutoControlConfig(groups=[]),
            )
        )
        # if not live_data.configure():
        #     raise RuntimeError("Failed to configure the interactor environment.")
        live_data.reset()
        self.live_data = live_data
        action_mode = (
            SystemMode.SAMPLING
            if self._shared_config.data_loader.future_span <= 1
            else SystemMode.RESETTING
        )
        self.get_logger().info(f"Action mode set to {action_mode}.")
        self._action = GroupsSendActionConfig(
            groups=["/"],
            action_values=[[]],
            modes=[action_mode],
        )

    def add_first_batch(self, batch: DictBatch):
        print(f"{list(batch.keys())=}")
        if self._shared_config.stage == "infer":
            if (length := len(batch["cur_state"])) != 1:
                raise ValueError(
                    f"Interactor only supports batch size of 1. Got {length}."
                )
        if batch:
            reset_action = GroupsSendActionConfig(
                groups=["/"],
                action_values=[batch["cur_state"][0][0].tolist()],
                modes=[SystemMode.RESETTING],
            )
            self.live_data.write(reset_action)

    def _send_action(self, action: list):
        self._action.action_values[0] = action
        self.live_data.write(self._action)
        _ = yield

    def _send_actions(self, actions: List[list]):
        for action in actions:
            yield from self._send_action(action)

    def _set_model_output(self, pred: torch.Tensor):
        # TODO: should we unify the pred to be BTD?
        if len(pred.shape) == 2:
            func = self._send_action
        else:
            func = self._send_actions
        return func(pred[0].tolist())

    def _set_batch(self, batch: DictBatch):
        return self._send_actions(batch["next_state"][0].tolist())

    def interact(self, value):
        # first request for the model prediction
        prediction, batch = yield ((YieldKey.PREDICT, self._get_model_input(value)),)
        # then send actions based on the prediction and batches
        if self.config.action_from == "data_loader":
            yield from self._set_batch(batch)
        elif self.config.action_from == "model":
            yield from self._set_model_output(prediction)

    def get_env_data(self) -> DictBatch:
        return self._dic_map(self.live_data.read())

    def _get_model_input(self, value: SendValue) -> DictBatch:
        # print(f"{batch['cur_state']=}, {batch['next_state']=}")
        last_prediction, batch = value
        data = (
            batch
            if self.config.model_input_from == "data_loader"
            else self._env_batcher([self.get_env_data()])
        )
        if self.config.open_loop_predict and last_prediction is not None:
            data["cur_state"] = last_prediction
        if self.use_extractor:
            features = {}
            for from_key, to_key in zip(self.from_keys, self.to_keys):
                raw_image = data[from_key][0]
                # cv2.imwrite("raw_image.png", raw_image)
                # assert not isinstance(raw_image, torch.Tensor)
                features[to_key] = {
                    "data": self.extractor.process_image(
                        raw_image[:, :, ::-1].copy(), self.config.extractor.prompt
                    )["features_proj"].squeeze(0)
                }
            # print(f"{features.keys()=}")
            batched_features = self._extra_batcher([features])
            return ChainMap(batched_features, data)
        return data

    def shutdown(self):
        return self.live_data.close()
