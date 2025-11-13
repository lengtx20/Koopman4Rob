from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    NonNegativeFloat,
    ConfigDict,
    Field,
)
from typing import List, Optional, Literal, Any, Set, Dict, Union, Tuple
from collections.abc import Callable
from functools import cache
from pathlib import Path
from mcap_data_loader.pipelines import HorizonConfig, PairWiseConfig
from interactor import InteractorBasis
from basis import ModelLike
from mcap_data_loader.callers.stack import StackType
from mcap_data_loader.callers.dict_tuple import DictTupleConfig
from mcap_data_loader.datasets.dataset import IterableMultiEpisodeDatasetsProtocol


IterUnit = Literal["epoch", "step", "sample", "minute"]
MODEL_CONFIG = ConfigDict(validate_assignment=True, extra="forbid")


class ParallelConfig(BaseModel):
    """Configuration for parallel processing.
    The data will be processed in parallel either in num_workers threads or
    processes. At most one iter() is created from source, and at most one
    thread will call next() on it at once.
    """

    model_config = MODEL_CONFIG

    num_workers: NonNegativeInt = 0
    """The number of workers to use for parallel processing. 0 means no parallelism."""
    in_order: bool = True
    """Whether to return items in the order from which they arrive from. If true, the iterator will return items in the order from which they arrive from source's iterator, potentially blocking even if other items are available."""
    method: Literal["thread", "process"] = "thread"
    """The method to use for parallel processing."""
    multiprocessing_context: Optional[Literal["spawn", "forkserver", "fork"]] = None
    """The multiprocessing context to use for parallel processing. 
    If None, the OS default context will be used."""
    max_concurrent: Optional[PositiveInt] = None
    """At most number of items will be either processed or in the iterator's output queue, 
    to limit CPU and Memory utilization. If None (default) the value will be 2 * num_workers."""
    snapshot_frequency: NonNegativeInt = 1
    """Frequency (in number of samples) to take snapshots of the parallel processing."""


class DataLoaderConfig(BaseModel):
    """Configuration for data loader."""

    model_config = MODEL_CONFIG

    stack: StackType = {}
    """Stack the dict values of a list of keys into a single value with the given key."""
    weights: List[str] = []
    """List of weight keys for each episode."""
    batch_size: PositiveInt = 1
    """Batch size for data loading."""
    drop_last: bool = False
    """Whether to drop the last incomplete batch."""
    num_workers: int = 0
    """Number of workers for data loading. 0 means single-threaded."""
    horizon: Optional[HorizonConfig] = None
    """Configuration for horizon processing."""
    pairwise: Optional[PairWiseConfig] = None
    """Configuration for pairwise processing."""
    dict_tuple: DictTupleConfig = DictTupleConfig()
    """Configuration for dict tuple processing."""
    parallel: ParallelConfig = ParallelConfig()
    """Configuration for parallel processing."""
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

    stage: Literal["train", "test", "infer"]
    """Stage of the task."""
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
    loss_fn: Callable[[Any, Any], Any]
    """Loss function to use."""
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
    """Configuration for testing."""

    save_results: bool = True
    """Whether to save the test results."""
    show_plot: bool = False
    """Whether to show plots of the results."""
    rollout_steps: PositiveInt = 1
    """Number of steps for rollout prediction (>=1)."""


class KeyConditionConfig(BaseModel):
    """Configuration for specifying stopping criteria based on keys."""

    necessary: Set[str] = set()
    """Set of keys that are necessary conditions for stopping."""
    sufficient: Set[str] = set()
    """Set of keys that are sufficient conditions for stopping."""

    @property
    def keys(self) -> Set[str]:
        return self.necessary | self.sufficient


class IntermittentConfig(BaseModel):
    """Configuration for interval-based operations."""

    unit: IterUnit = "minute"
    """Unit of the interval."""
    interval: PositiveInt = 1
    """Interval in the specified unit."""
    maximum: NonNegativeInt = 0
    """Maximum number of times to perform the operation. If set to 0, there is no limit."""


class SnapshotConfig(IntermittentConfig):
    """Configuration for snapshot taking."""

    keys: Set[str] = set()
    """Set of metric keys to include in the snapshot."""


class TrainIterationConfig(BaseModel):
    """Configuration for training iteration.

    This class defines various stopping and continuation criteria for a training loop.
    Any criterion set to its default "no limit" value (typically 0 or 0.0) is effectively disabled.
    Training will stop when any of the sufficient conditions are met or all necessary conditions are satisfied.
    """

    model_config = MODEL_CONFIG

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
    """Configuration for training."""

    model_config = MODEL_CONFIG

    task_id: PositiveInt = 1
    """Identifier for the training task."""
    ewc_threshold: float = 1.0
    """Threshold value for EWC regularization."""
    ewc_regularization: bool = False
    """Whether to apply EWC regularization during training."""
    iteration: TrainIterationConfig = Field(default_factory=TrainIterationConfig)
    """Configuration for training iteration."""
    snapshot: List[SnapshotConfig] = []
    """List of snapshot configurations."""
    save_model: SaveModelConfig = SaveModelConfig()
    """Configuration for saving the model."""
    train_val_split: Union[float, Tuple[int, int]] = 0.8
    """Train/validation datasets split configuration.
    If float, represents the proportion of data to use for training.
    If tuple, represents the (train_step, val_step). This means that `train_step` data points are taken from the `episode` list, followed by `val_step` data points, and so on. The final ratio of the training set to the validation set is approximately `train_step:val_step`. This method is suitable for repeatedly collecting data N times under the same settings, then changing to the next setting and continuing to collect data. After traversing M settings, there are a total of N * M episodes. During training, this partitioning method ensures that the validation set covers all different settings.
    """


class InferConfig(BaseModel):
    """Configuration for inference."""

    model_config = MODEL_CONFIG

    max_rollouts: NonNegativeInt = 0
    """Maximum number of rollouts to perform during inference."""
    max_steps: NonNegativeInt = 0
    """Maximum number of steps to perform during inference."""
    frequency: int = 0
    """The frequency (in steps) to send action commands.
    0 means wait for any input after every step. Negative means no limit.
    """
    rollout_wait: Any = "input"


class Config(CommonConfig):
    """Main configuration"""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", arbitrary_types_allowed=True
    )

    datasets: IterableMultiEpisodeDatasetsProtocol
    """Datasets for training/testing/inference."""
    data_loader: DataLoaderConfig
    """Configuration for the data loader."""
    model: ModelLike
    """Configuration for the model."""
    train: TrainConfig = TrainConfig()
    """Configuration for training."""
    test: TestConfig = TestConfig()
    """Configuration for testing."""
    infer: InferConfig = InferConfig()
    """Configuration for inference."""
    interactor: Optional[InteractorBasis] = None
    """Configuration for the interactor."""
    extra: Dict[str, Any] = {}
    """Extra configuration parameters. It is useful 
    to store intermediate parameters in this field in the
    hydra config file to avoid extra forbid error."""
