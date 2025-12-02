from airbot_data_collection.common.systems.basis import SystemMode
from airbot_data_collection.common.live_data.grouped import (
    GroupedSystemDataSource,
    GroupsSendActionConfig,
)
from mcap_data_loader.utils.basic import remove_util
from mcap_data_loader.callers.dict_map import DictMap, DictMapConfig
from mcap_data_loader.callers.stack import BatchStacker, BatchStackerConfig, DictBatch
from pydantic import BaseModel
from config import Config, ConfigDict, Stage
from collections import defaultdict
from data.blip2_feature_extractor import Blip2ImageFeatureExtractor
from interactor import InteractorBasis, YieldKey, SendValue
from pathlib import Path
from pprint import pformat
from typing import Literal, List
from more_itertools import collapse
from datetime import datetime
import torch
import cv2
import json


class ExtractorConfig(BaseModel, frozen=True):
    """Configuration for the feature extractor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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


class InteractorConfig(BaseModel, frozen=True):
    """Configuration for the interactor between the model and the environment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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
    save_image: bool = False
    """Whether to save images during inference."""
    show_image: bool = False
    """Whether to display images during inference."""
    # image_transform: bool = False
    # """Whether to apply image transformations during inference."""
    stack: BatchStackerConfig
    """Stacking configuration for the interactor."""
    action_mode: Literal["sampling", "resetting"] = "resetting"
    """Mode for sending actions to the environment."""
    # reset_action: List[float]
    """"""


class Interactor(InteractorBasis):
    config: InteractorConfig

    def _remove_prefix(self, keys: List[str]) -> List[str]:
        return [remove_util(key, "/", True) for key in keys]

    def add_config(self, config: Config):
        self._save_root = Path(f"logs/infer/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._save_root.mkdir(parents=True, exist_ok=True)
        dl_stack_cfg = self.config.stack
        stack_dl = dl_stack_cfg.stack
        stack_env = {}
        for cat_key, keys in stack_dl.items():
            # remove possible next cat key
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
                dl_stack_cfg.model_copy(update={"stack": torch_stack})
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
            dl_stack_cfg.model_copy(update={"stack": stack_env, "backend_out": "torch"})
        )
        self._shared_config = config
        self._env_keys = set(collapse(stack_env.values()))
        self._live_dict_map = DictMap(
            DictMapConfig(
                dtype=dl_stack_cfg.dtype,
                device=dl_stack_cfg.device,
                # TODO: torch will be much faster
                backend_out="torch",
                keys_include=self._env_keys,
                replace=True,
            )
        )
        self._init_live_data()
        self._rollout = -1

    def _init_live_data(self):
        live_data: GroupedSystemDataSource = self._shared_config.data_loaders.get(
            "live", None
        )
        if live_data is not None:
            # if not live_data.configure():
            #     raise RuntimeError("Failed to configure the interactor environment.")
            live_data.reset()
        self.live_data = live_data
        action_mode = SystemMode[self.config.action_mode.upper()]
        self.get_logger().info(f"Action mode set to {action_mode}.")
        self._action = GroupsSendActionConfig(
            groups=["/"],
            action_values=[[]],
            modes=[action_mode],
        )
        self._recorder = None

    def add_first_batch(self, batch: DictBatch):
        if batch:
            self.get_logger().info(f"{batch.keys()=}")
            self.get_logger().info(f"{batch['cur_state']=}")
            if self._shared_config.stage is Stage.INFER:
                if (length := len(batch["cur_state"])) != 1:
                    raise ValueError(
                        f"Interactor only supports batch size of 1. Got {length}."
                    )
            fixed_value = [0.7]
            reset_action = GroupsSendActionConfig(
                groups=["/"],
                action_values=[batch["cur_state"][0][0].tolist() + fixed_value],
                modes=[SystemMode.RESETTING],
            )
            if self.live_data is not None:
                self.live_data.write(reset_action)
        if self._recorder is not None:
            path = self._save_root / f"{self._rollout}/recordings.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(self._recorder, open(path, "w"), indent=4)
            self.get_logger().info(f"Saved recordings to {path}.")
        self._recorder = defaultdict(list)
        self._rollout += 1
        self._step = -1

    def _send_actions(self, actions: List[list]):
        for action in actions:
            self.get_logger().info(f"sending action: {action}")
            self._action.action_values[0] = action
            self.live_data.write(self._action)
            _ = yield

    def _set_model_output(self, pred: torch.Tensor):
        # TODO: should we unify the pred to be BTD?
        if len(pred.shape) != 3:
            raise ValueError(
                f"Expected prediction shape to be 3 (BTD), got {pred.shape}"
            )
        elif pred.shape[0] != 1:
            raise ValueError(
                f"Expected batch size of 1 for prediction, got {pred.shape[0]}"
            )
        return self._send_actions(pred.squeeze(0).tolist())

    def _set_batch(self, batch: DictBatch):
        return self._send_actions(batch["next_state"][0].tolist())

    def interact(self, value):
        # first request for the model prediction
        prediction, batch = yield ((YieldKey.PREDICT, self._get_model_input(value)),)
        # print(f"{prediction.shape=}")
        # then send actions based on the prediction and batches
        if self.config.action_from == "data_loader":
            yield from self._set_batch(batch)
        elif self.config.action_from == "model":
            yield from self._set_model_output(prediction)

    def get_env_data(self) -> DictBatch:
        return self._live_dict_map(self.live_data.read())

    def _get_model_input(self, value: SendValue) -> DictBatch:
        # print(f"{batch['cur_state']=}, {batch['next_state']=}")
        self._step += 1
        last_prediction, batch = value
        config = self.config
        use_batch = config.model_input_from == "data_loader"
        data = batch if use_batch else self._env_batcher([self.get_env_data()])
        if config.open_loop_predict and last_prediction is not None:
            data["cur_state"] = last_prediction
        if self.use_extractor:
            features = {}
            for from_key, to_key in zip(self.from_keys, self.to_keys):
                raw_image = data[from_key][0]
                if use_batch:
                    rgb_image = raw_image
                    bgr_image = raw_image[:, :, ::-1].copy()
                else:
                    rgb_image = raw_image[:, :, ::-1].copy()
                    bgr_image = raw_image
                rollout_root = self._save_root / str(self._rollout)
                image_rela_path = (
                    f"{from_key.removeprefix('/').replace('/', '.')}/{self._step}.png"
                )
                path = rollout_root / image_rela_path
                path.parent.mkdir(parents=True, exist_ok=True)
                if config.save_image:
                    cv2.imwrite(path, bgr_image)
                    if not use_batch:  # save batch images as well for test
                        batch_path = rollout_root / "batch" / image_rela_path
                        batch_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(batch_path, batch[from_key][0][:, :, ::-1])
                    self.get_logger().info(f"Saved image to {path}.")
                if config.show_image:
                    cv2.imshow(from_key, bgr_image)
                # assert not isinstance(raw_image, torch.Tensor)
                features[to_key] = {
                    "data": self.extractor.process_image(
                        rgb_image, config.extractor.prompt
                    )["features_proj"].squeeze(0)
                }
                self._recorder[to_key].append(features[to_key]["data"].tolist())
            # print(f"{features.keys()=}")
            # print(f"{batch['cur_action'].norm()=}")
            batched_features = self._extra_batcher([features])
            # compare features
            self.get_logger().info(
                f"features diff: {(batched_features['cur_action'] - batch['cur_action']).norm()}"
            )
            data.update(batched_features)
            # print(f"{data.keys()=}")
            self.get_logger().info(f"{data['cur_state']=}")
            return data
        return data

    def shutdown(self):
        if self.live_data is not None:
            return self.live_data.close()
