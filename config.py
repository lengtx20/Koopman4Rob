from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    NonNegativeFloat,
    ConfigDict,
    Field,
)
from typing import List, Optional, Literal, Any, Set, Dict, Union, Tuple
from functools import cache
from pathlib import Path
from mcap_data_loader.utils.basic import NonIteratorIterable


IterUnit = Literal["epoch", "step", "sample", "minute"]


class DataLoaderConfig(BaseModel):
    """Configuration for data loader."""

    states: List[str] = Field(min_length=1)
    """List of state keys."""
    actions: List[str] = Field(min_length=1)
    """List of action keys."""
    weights: List[str] = []
    """List of weight keys for each episode."""
    batch_size: PositiveInt = 1
    """Batch size for data loading."""
    num_workers: int = 0
    """Number of workers for data loading. 0 means single-threaded."""
    pair_gap: NonNegativeInt = 0
    """Gap between paired samples in epairwise."""
    prefetch_factor: NonNegativeInt = 0
    """Number of samples to prefetch per worker."""
    prefetch_snapshot_frequency: NonNegativeInt = 1
    """Frequency (in number of samples) to take snapshots of the data loader."""
    pin_memory_device: Optional[str] = None
    """Device to pin memory to. If None, pin memory is not used. If set to "", defaults to "cuda" if available."""
    pin_memory_snapshot_frequency: NonNegativeInt = 1
    """Frequency (in number of samples) to take snapshots when using pin memory."""
    restart_on_stop_iteration: bool = True
    """Whether to restart the data loader when StopIteration is encountered."""
    normalize: bool = False
    """Whether to normalize the data."""


class CommonConfig(BaseModel):
    """Common configuration parameters.
    These configurations are applicable to both training and testing
    """

    model_config = ConfigDict(validate_assignment=True)

    mode: Literal["train", "test", "infer"]
    """Mode of stage: 'train', 'test', or 'infer'."""
    root_dir: Path = Path("logs")
    """Root directory for saving logs and checkpoints."""
    checkpoints_dir: Path = Path("checkpoints")
    """Directory for saving model checkpoints."""
    checkpoint_path: Optional[Path] = Path("0")
    """Path to a specific model checkpoint. If None, no checkpoint will be saved."""
    seed: int = 42
    """Random seed for reproducibility."""
    device: str = ""
    """Device to load tensors onto. If empty, will use "cuda" if available else "cpu"."""
    dtype: str = "float32"
    """Data type for tensors. E.g., 'float32', 'float16'."""
    tb_log_dir: Path = Path("tensorboard")
    ewc_model: Optional[Path] = None
    ewc_lambda: float = 100.0

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


class IntermittentConfig(BaseModel):
    """Configuration for interval-based operations.
    Args:
        unit (IterUnit): Unit of the interval.
        interval (PositiveInt): Interval in the specified unit.
        maximum (NonNegativeInt): Maximum number of times to perform the operation. If set to 0 (default), there is no limit.
    """

    unit: IterUnit = "minute"
    interval: PositiveInt = 1
    maximum: NonNegativeInt = 0


class SnapshotConfig(IntermittentConfig):
    """Configuration for snapshot saving.

    This class defines the configuration for saving snapshots during training.
    Args:
        keys (Set[str]): Set of metric keys to include in the snapshot.
        unit (IterUnit): Unit of the interval for saving snapshots.
        interval (PositiveInt): Interval for saving snapshots in the specified unit.
        maximum (NonNegativeInt): Maximum number of snapshots to keep. If set to 0
            (default), all snapshots are kept.
    """

    keys: Set[str] = set()


class TrainIterationConfig(BaseModel):
    """Configuration for training iteration.

    This class defines various stopping and continuation criteria for a training loop.
    Any criterion set to its default "no limit" value (typically 0 or 0.0) is effectively disabled.
    Training will stop when any of the sufficient conditions are met or all necessary conditions are satisfied.
    """

    patience: NonNegativeInt = 0
    """Number of consecutive epochs with no improvement in the train or val loss before early stopping is triggered.
    Set to 0 to disable early stopping."""
    max_epoch: NonNegativeInt = 0
    """Maximum number of epochs allowed for training."""
    min_epoch: NonNegativeInt = 0
    """Minimum number of epochs that must be completed before any stopping
    condition (e.g., patience or loss thresholds) is evaluated."""
    max_step: NonNegativeInt = 0
    """Maximum number of training steps (batches processed) allowed."""
    min_step: NonNegativeInt = 0
    """Minimum number of training steps that must be completed before stopping
    conditions are evaluated."""
    max_sample: NonNegativeInt = 0
    """Maximum number of training samples processed (across all epochs)."""
    min_sample: NonNegativeInt = 0
    """Minimum number of training samples that must be processed before stopping
    conditions are evaluated."""
    max_time: NonNegativeFloat = 0.0
    """Maximum wall-clock training and validation time in minutes. Training stops once exceeded.
    Set to 0.0 for no time limit."""
    min_time: NonNegativeFloat = 0.0
    """Minimum training and validation time in minutes that must elapse before any stopping
    condition is considered."""
    max_train_time: NonNegativeFloat = 0.0
    """Maximum training time in minutes (excluding validation). Training stops once exceeded."""
    min_train_time: NonNegativeFloat = 0.0
    """Minimum training time in minutes that must elapse before any stopping condition is considered."""
    max_train_loss: NonNegativeFloat = 0.0
    """Upper bound on training loss; training stops if loss exceeds this value."""
    min_train_loss: NonNegativeFloat = 0.0
    """Lower bound on training loss; training will not stop if loss drops below this value."""
    max_val_loss: NonNegativeFloat = 0.0
    """Upper bound on validation loss; training stops if validation loss exceeds this value."""
    min_val_loss: NonNegativeFloat = 0.0
    """Lower bound on validation loss; training will not stop if validation loss drops below this value."""
    conditions: KeyConditionConfig = KeyConditionConfig()
    """Configuration for stopping conditions."""

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


class SaveModelConfig(BaseModel):
    """Configuration for saving the model."""

    period: Optional[IntermittentConfig] = None
    """Configuration for periodic saving of the model."""
    on_improve: List[Literal["train_loss", "val_loss"]] = ["val_loss"]
    """Saves the model when the specified metrics improve."""
    maximum: List[NonNegativeInt] = [5]
    """Maximum number of saved models for each metric in on_improve."""

    def model_post_init(self, context):
        if len(self.on_improve) != len(self.maximum):
            raise ValueError("Length of on_improve and maximum must be the same")


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
    save_model: SaveModelConfig = SaveModelConfig()
    train_val_split: Union[float, Tuple[int, int]] = 0.8


class InferConfig(BaseModel):
    """Configuration for inference."""

    # TODO: use deriving structure for config
    extra_models: Dict[str, Dict] = {}
    """paths to extra models, e.g. vision backbones, for inference"""
    obs_from_dataset: bool = False
    """Whether to use observations from the dataset during inference."""
    action_from_dataset: bool = False
    """Whether to use actions from the dataset during inference."""
    open_loop_predict: bool = False
    """Whether to perform open-loop prediction during inference."""
    send_action: bool = True
    """Whether to send action commands during inference."""
    max_rollouts: NonNegativeInt = 0
    """Maximum number of rollouts to perform during inference."""
    max_steps: NonNegativeInt = 0
    """Maximum number of steps to perform during inference."""
    frequency: int = 0
    """The frequency (in steps) to send action commands.
    0 means wait for input after every step. Negative means no limit.
    """
    show_image: bool = False
    """Whether to display images during inference."""
    image_transform: bool = False
    """Whether to apply image transformations during inference."""
    feature_from_dataset: bool = True
    """Whether to use features from the dataset during inference."""


class ModelConfig(BaseModel):
    """Configuration for the model."""

    state_dim: NonNegativeInt = 0
    """Dimension of the system state. If 0, will be inferred from data."""
    action_dim: NonNegativeInt = 0
    """Dimension of the control input. If 0, will be inferred from data."""
    hidden_sizes: List[PositiveInt]
    """Sizes of hidden layers in the model."""
    lifted_dim: PositiveInt
    """Dimension of the lifted space."""
    activation: str = "relu"
    """Activation function to use in the model."""
    include_iden_state: bool = True
    """Whether to include identity state in the model."""
    iden_decoder: bool = True
    """Whether to use identity decoder in the model."""


class Config(CommonConfig):
    """Main configuration"""

    datasets: List[NonIteratorIterable] = Field(min_length=1)
    """Configuration for the dataset."""
    data_loader: DataLoaderConfig
    """Configuration for the data loader."""
    model: ModelConfig
    """Configuration for the model."""
    train: TrainConfig = TrainConfig(
        iteration=TrainIterationConfig(
            max_train_time=20,
            patience=250,
            max_epoch=5000,
            min_val_loss=0.00045,
            # ensure at least 20 minutes training
            min_train_time=20,
        ),
        snapshot=[
            SnapshotConfig(
                keys={"train_loss", "val_loss", "min_train_loss", "min_val_loss"},
                interval=1,
            ),
        ],
        train_val_split=(2, 1),
    )
    """Configuration for training."""
    test: TestConfig = TestConfig()
    """Configuration for testing."""
    infer: InferConfig = InferConfig(
        extra_models={
            "blip2-itm-vit-g": {
                "path": Path("pretrained_models/blip2-itm-vit-g"),
                "prompt": "The end effector of the robotic arm tries to get close to the QR code attached to the cabinet.",
            }
        },
        frequency=0,
        show_image=False,
        # test for one step prediction of the dataset
        # obs_from_dataset=True,
        # action_from_dataset=False,
        # feature_from_dataset=False,
        # test for realtime continuous infering
        obs_from_dataset=False,
        feature_from_dataset=False,
        action_from_dataset=False,
        open_loop_predict=True,
    )
    """Configuration for inference."""
