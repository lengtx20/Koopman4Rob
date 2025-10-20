from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    NonNegativeFloat,
    Field,
)
from typing import List, Optional, Literal, Any, Set
from functools import cache
from pathlib import Path


cam_keys = ["/env_camera/color/image_raw"]
feature_suffix = "features_proj"


class CommonConfig(BaseModel):
    """Common configuration parameters.
    These configurations are applicable to both training and testing
    but the values are not necessarily the same in both cases.
    Args:
        root_dir: Path, root directory for saving logs and checkpoints.
        checkpoints_dir: Path, directory for saving model checkpoints.
        checkpoint_path: Optional[Path], path to a specific model checkpoint.
            If None, no checkpoint will be saved.
        data_dir: Path, path to the data directory.
        seed: int, random seed for reproducibility.
        batch_size: PositiveInt, batch size for training/testing.
        num_workers: int, number of workers for data loading.
        mode: Literal["test", "train"], mode of stage.
        img_features_keys: List[str], list of keys for image features.
        ewc_model: Optional[Path], path to the EWC model checkpoint.
        ewc_lambda: float, regularization strength for EWC.
        tb_log_dir: Path, directory for TensorBoard logs.
        device: str, device to use for computation (e.g., "cpu", "cuda:0").
    """

    mode: Literal["train", "test"]
    data_dir: Path
    root_dir: Path = Path("logs")
    checkpoints_dir: Path = Path("checkpoints")
    checkpoint_path: Optional[Path] = Path("0")
    seed: int = 42
    batch_size: PositiveInt = 64
    num_workers: int = 0
    robot_state_keys: List[str] = []
    robot_action_keys: List[str] = [
        "/follow/arm/joint_state/position",
        # "/follow/eef/joint_state/position",
    ]
    img_features_keys: List[str] = [f"{cam}/{feature_suffix}" for cam in cam_keys]
    pair_gap: NonNegativeInt = 0
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
            return path if (path.is_absolute() or path.exists()) else relative_to / path

        self.checkpoints_dir = process_path(self.checkpoints_dir)
        self.checkpoint_path = process_path(self.checkpoint_path, self.checkpoints_dir)
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
    """Configuration for specifying stopping criteria based on keys.
    Args:
        necessary: Set[str], set of keys that are necessary conditions for stopping.
        sufficient: Set[str], set of keys that are sufficient conditions for stopping.
    """

    necessary: Set[str] = set()
    sufficient: Set[str] = set()

    @property
    def keys(self) -> Set[str]:
        return self.necessary | self.sufficient


class SnapshotConfig(BaseModel):
    """Configuration for snapshot saving."""

    keys: Set[str] = set()
    unit: Literal["epoch", "step", "sample", "minute"] = "minute"
    interval: PositiveInt = 1
    maximum: NonNegativeInt = 0


class TrainIterationConfig(BaseModel):
    """Configuration for training iteration.

    This class defines various stopping and continuation criteria for a training loop.
    Any criterion set to its default "no limit" value (typically 0 or 0.0) is effectively disabled.
    Training will stop when any of the sufficient conditions are met or all necessary conditions are satisfied.

    Args:
        patience (NonNegativeInt): Number of consecutive epochs with no improvement in the monitored
            metric before early stopping is triggered. Set to 0 to disable early stopping.
        max_epoch (NonNegativeInt): Maximum number of epochs allowed for training.
        min_epoch (NonNegativeInt): Minimum number of epochs that must be completed before any stopping
            condition (e.g., patience or loss thresholds) is evaluated.
        max_step (NonNegativeInt): Maximum number of training steps (batches processed) allowed.
        min_step (NonNegativeInt): Minimum number of training steps that must be completed before stopping
            conditions are evaluated.
        max_sample (NonNegativeInt): Maximum number of training samples processed (across all epochs).
        min_sample (NonNegativeInt): Minimum number of training samples that must be processed before stopping
            conditions are evaluated.
        max_time (NonNegativeFloat): Maximum wall-clock training time in minutes. Training stops once exceeded.
            Set to 0.0 for no time limit.
        min_time (NonNegativeFloat): Minimum training time in minutes that must elapse before any stopping
            condition is considered.
        max_train_loss (NonNegativeFloat): Upper bound on training loss; training stops if loss exceeds this value.
        min_train_loss (NonNegativeFloat): Lower bound on training loss; training stops once loss drops below this value.
        max_val_loss (NonNegativeFloat): Upper bound on validation loss; training stops if validation loss exceeds this value.
        min_val_loss (NonNegativeFloat): Lower bound on validation loss; training stops once validation loss drops below this value.
        conditions (KeyConditionConfig): Additional structured stopping conditions, typically mapping metric names
            to comparison criteria (e.g., "val_acc >= 0.95"). Invalid or unsupported metric keys will raise a ValueError.

    Raises:
        ValueError: If invalid keys are provided in the `conditions` configuration.
    """

    patience: NonNegativeInt = 0
    max_epoch: NonNegativeInt = 0
    min_epoch: NonNegativeInt = 0
    max_step: NonNegativeInt = 0
    min_step: NonNegativeInt = 0
    max_sample: NonNegativeInt = 0
    min_sample: NonNegativeInt = 0
    max_time: NonNegativeFloat = 0.0
    min_time: NonNegativeFloat = 0.0
    max_train_loss: NonNegativeFloat = 0.0
    min_train_loss: NonNegativeFloat = 0.0
    max_val_loss: NonNegativeFloat = 0.0
    min_val_loss: NonNegativeFloat = 0.0
    conditions: KeyConditionConfig = KeyConditionConfig()

    def model_post_init(self, context):
        valid_keys = self.get_valid_keys()
        if self.conditions.keys:
            invalid_keys = self.conditions.keys - valid_keys
            if invalid_keys:
                raise ValueError(f"Invalid condition keys: {invalid_keys}")
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
        task_id: PositiveInt, identifier for the training task.
        fisher_path: Optional[Path], path to save/load Fisher information matrix for EWC.
        threshold_mode: Optional[Literal["neural_ratio", "ewc_loss"]], mode for EWC thresholding.
        ewc_threshold: float, threshold value for EWC regularization.
        ewc_regularization: bool, whether to apply EWC regularization during training.
        loss_fn: Any, loss function to use during training.
        iteration: TrainIterationConfig, configuration for training iteration.
        snapshot: List[SnapshotConfig], list of snapshot configurations.
    """

    task_id: PositiveInt = 1
    fisher_path: Optional[Path] = None
    threshold_mode: Optional[Literal["neural_ratio", "ewc_loss"]] = None
    ewc_threshold: float = 1.0
    ewc_regularization: bool = False
    loss_fn: Any = "MSELoss"
    iteration: TrainIterationConfig = Field(default_factory=TrainIterationConfig)
    snapshot: List[SnapshotConfig] = []


class ModelConfig(BaseModel):
    """Configuration for the model.
    Args:
        hidden_sizes: List[int], list of hidden layer sizes.
        lifted_dim: PositiveInt, dimension of the lifted space (>=1).
    """

    state_dim: int = 0
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
    train: TrainConfig = TrainConfig(
        iteration=TrainIterationConfig(
            max_time=20,
            patience=250,
            max_epoch=500,
            min_val_loss=0.00045,
            # ensure at least 20 minutes training
            min_time=20,
        ),
        snapshot=[
            SnapshotConfig(
                keys={"train_loss", "val_loss", "min_train_loss", "min_val_loss"},
                interval=1,
            ),
        ],
    )
    test: TestConfig = TestConfig()

    def model_post_init(self, context):
        super().model_post_init(context)
        # TODO: use warming up to determine the state_dim and action_dim if 0
        state_dim = 6 if len(self.robot_action_keys) == 1 else 7
        if self.model.state_dim == 0:
            self.model.state_dim = state_dim
        else:
            assert self.model.state_dim == state_dim, (
                f"Configured state_dim {self.model.state_dim} does not match "
                f"the data state_dim {state_dim}"
            )
