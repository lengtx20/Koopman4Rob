from typing import runtime_checkable, Union, List, Protocol, Optional, Any
from pathlib import Path
from typing_extensions import Self
from collections import ChainMap
from airbot_data_collection.common.utils.array_like import Array


DictBatch = ChainMap[str, Union[Array, List[Array], int]]


@runtime_checkable
class ModelLike(Protocol):
    """Protocol for model-like objects."""

    def add_first_batch(self, batch: DictBatch) -> None: ...
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def load(self, path: Optional[Path] = None) -> Self: ...
    def save(self, path: Path) -> None: ...
    def parameters(self) -> Any: ...
