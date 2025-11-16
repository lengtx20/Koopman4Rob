from abc import abstractmethod
from mcap_data_loader.basis.cfgable import ConfigurableBasis
from collections.abc import Generator
from typing import Any, Sequence, Union, Optional, NamedTuple, Tuple, TYPE_CHECKING
from enum import Enum, auto
from basis import DictBatch


if TYPE_CHECKING:
    # circular import, so we only import during type checking
    from config import Config
else:
    Config = Any


class ReturnAction(Enum):
    """Possible return values for the interactor update generator."""

    NEXT = auto()
    """Break to the next rollout."""
    EXIT = auto()
    """Exit the entire process."""
    WAIT = auto()
    """Pause the interactor for external handling."""
    REPEAT = auto()
    """Break and repeat this rollout."""


class YieldKey(Enum):
    """Possible yielded values from the interactor update generator."""

    NEXT_BATCH = auto()
    """Request the next batch from the data loader."""
    PREDICT = auto()
    """Request the model to make a prediction."""


class YieldItem(NamedTuple):
    """Item yielded from the interactor update generator."""

    key: YieldKey
    value: Optional[Any] = None


SendValue = Tuple[Any, DictBatch]
YieldValue = Optional[Sequence[YieldItem]]
ReturnValue = Optional[Union[Any, ReturnAction]]


class InteractorBasis(ConfigurableBasis):
    def on_configure(self):
        return True

    @abstractmethod
    def add_config(self, config: Config):
        """Add the global shared configuration to the interactor."""

    @abstractmethod
    def add_first_batch(self, batch: DictBatch):
        """Add the first batch of data from the data loader to the interactor."""

    @abstractmethod
    def interact(
        self, value: SendValue
    ) -> Generator[YieldValue, SendValue, ReturnValue]:
        """Interact with the outside model and data loader based on a generator.
        This allows the interactor to both support loop calls internally and delegate
        calling authority externally, such as controlling the loop frequency. Furthermore,
        the generator's flexible bidirectional communication mechanism allows interaction
        at each step of the loop.
        """

    @abstractmethod
    def shutdown(self):
        """Shutdown the interactor."""
