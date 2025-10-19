from pydantic import BaseModel, PositiveInt, NonNegativeInt, NonNegativeFloat, Field
from typing import List, Optional, Literal, Any, Set
from functools import cache, cached_property
from pathlib import Path


cam_keys = ["/env_camera/color/image_raw"]
feature_suffix = "features_proj"


class CommonConfig(BaseModel):
    """Common configuration parameters.
    These configurations are applicable to both training and testing
    but the values are not necessarily the same in both cases.
    Args:
        root_dir: Path, root directory for saving logs and checkpoints.
        checkpoints_dir: Optional[Path], directory for saving model checkpoints.
            If None, checkpoints will not be saved.
        data_dir: str, path to the data directory.
        model_dir: Path, path to the model directory.
        fisher_path: Optional[str], path to the Fisher information matrix file.
        mode: Literal["test", "train"], mode of operation.
        img_features_keys: List[str], list of keys for image features.
    """

    root_dir: Path = Path("logs")
    checkpoints_dir: Path = Path("checkpoints")
    checkpoint_path: Optional[Path] = Path("0")
    data_dir: Path
    seed: int = 42
    batch_size: PositiveInt = 64
    num_workers: int = 0
    mode: Literal["test", "train"] = "test"
    img_features_keys: List[str] = [f"{cam}/{feature_suffix}" for cam in cam_keys]
    ewc_model: Optional[Path] = None
    ewc_lambda: float = 100.0
    tb_log_dir: Path = Path("tensorboard")
    device: str = ""

    def model_post_init(self, context):
        def process_path(
            path: Optional[Path], relative_to: Path = self.root_dir
        ) -> Optional[Path]:
            if path is None:
                return None
            return path if path.is_absolute() else relative_to / path

        self.checkpoints_dir = process_path(self.checkpoints_dir)
        self.checkpoint_path = process_path(self.checkpoint_path, self.checkpoints_dir)
        self.data_dir = process_path(self.data_dir)
        self.tb_log_dir = process_path(self.tb_log_dir)


class TestConfig(BaseModel):
    """Configuration for testing.
    Args:
        save_results: bool, whether to save the test results.
        show_plot: bool, whether to show plots of the results.
        rollout_steps: PositiveInt, number of steps for rollout prediction (>=1).
    """

    save_results: bool = True
    show_plot: bool = False
    rollout_steps: PositiveInt = 1


class KeyConditionConfig(BaseModel):
    necessary: Set[str] = set()
    sufficient: Set[str] = set()

    @cached_property
    def keys(self) -> Set[str]:
        return self.necessary | self.sufficient


class TrainIterationConfig(BaseModel):
    """Configuration for controlling training process.
    Args:
        iter_mode: Literal["epoch", "step", "sample"], mode of iteration control.
    """

    iter_mode: Literal["epoch", "step", "sample"] = "epoch"
    iter_max: NonNegativeInt = 0
    iter_min: NonNegativeInt = 0
    patience: NonNegativeInt = 0
    max_train_loss: NonNegativeFloat = 0.0
    min_train_loss: NonNegativeFloat = 0.0
    max_val_loss: NonNegativeFloat = 0.0
    min_val_loss: NonNegativeFloat = 0.0
    max_time: NonNegativeFloat = 0.0
    conditions: KeyConditionConfig = KeyConditionConfig()

    def model_post_init(self, context):
        valid_keys = self.get_valid_keys()
        if self.conditions.keys:
            invalid_keys = self.conditions.keys - valid_keys
            if invalid_keys:
                raise ValueError(f"Invalid priority keys: {invalid_keys}")
            min_in_suffi = self.conditions.sufficient & self.get_valid_keys("min")
            if min_in_suffi:
                raise ValueError(
                    f"min keys {min_in_suffi} can not be sufficient conditions"
                )
        else:
            for key in self.model_fields_set & valid_keys:
                if getattr(self, key) != 0:
                    if "min" in key:
                        self.conditions.necessary.add(key)
                    else:
                        self.conditions.sufficient.add(key)

    @classmethod
    @cache
    def get_valid_keys(cls, matching: str = "") -> Set[str]:
        keys = cls.model_fields.keys() - {"conditions", "iter_mode"}
        if matching:
            return {key for key in keys if matching in key}
        return keys


class TrainConfig(BaseModel):
    """Configuration for training.
    Args:
        max_epochs: PositiveInt, maximum number of training epochs (>=1).
        batch_size: PositiveInt, batch size for training (>=1).
    """

    task_id: PositiveInt = 1
    fisher_path: Optional[Path] = None
    threshold_mode: Optional[Literal["neural_ratio", "ewc_loss"]] = None
    ewc_threshold: float = 1.0
    ewc_regularization: bool = False
    loss_fn: Any = "MSELoss"
    iteration: TrainIterationConfig = Field(default_factory=TrainIterationConfig)


class ModelConfig(BaseModel):
    """Configuration for the model.
    Args:
        hidden_sizes: List[int], list of hidden layer sizes.
        lifted_dim: PositiveInt, dimension of the lifted space (>=1).
    """

    state_dim: int = 7
    action_dim: int = 256
    hidden_sizes: List[int] = [512, 512, 512]
    lifted_dim: PositiveInt = 256


class Config(CommonConfig):
    """Main configuration for the Koopman model.
    Args:
        model: ModelConfig, configuration for the model.
        train: TrainConfig, configuration for training.
        test: TestConfig, configuration for testing.
    """

    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    test: TestConfig = TestConfig()
