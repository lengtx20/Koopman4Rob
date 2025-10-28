from collections import deque
from pathlib import Path
from typing import Optional, Deque
import shutil


class FIFOSave:
    def __init__(self, maxlen: Optional[int] = None):
        self._deque: Deque[Path] = deque(maxlen=maxlen or None)

    def append(self, item: Path) -> Optional[Path]:
        if self._deque.maxlen is not None and len(self._deque) == self._deque.maxlen:
            to_remove = self._deque.popleft()
            if to_remove.exists():
                if to_remove.is_dir():
                    shutil.rmtree(to_remove)
                else:
                    to_remove.unlink()
        else:
            to_remove = None
        item.mkdir(parents=True, exist_ok=True)
        self._deque.append(item)
        return to_remove

    @property
    def last_item(self) -> Optional[Path]:
        if len(self._deque) == 0:
            return None
        return self._deque[-1]
