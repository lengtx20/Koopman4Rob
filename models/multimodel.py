from airbot_data_collection.common.callers.multi import MultiCaller, MultiCallerConfig
from airbot_data_collection.basis import ConfigurableBasis
from basis import ModelLike
from typing import Optional, List
from pathlib import Path


class MultiModelConfig(MultiCallerConfig[ModelLike]):
    paths: List[Path]


class MultiModel(ConfigurableBasis):
    config: MultiModelConfig
    interface: MultiCaller

    def on_configure(self):
        self.interface.configure()

    def add_first_batch(self, batch) -> None:
        for model in self.config.callables:
            model.add_first_batch(batch)
        if not self.configure():
            raise RuntimeError(
                "Failed to configure MultiModel after adding first batch"
            )

    def load(self, path: Optional[Path] = None):
        if path:
            self.get_logger().info(f"The given single path will be ignored: {path}")
            self.get_logger().info(f"Using paths from config: {self.config.paths}")
        for path, model in zip(self.config.paths, self.config.callables):
            model.load(path)

    def save(self, path: Optional[Path] = None):
        raise NotImplementedError("MultiModel save not implemented yet")

    def train(self):
        raise NotImplementedError("MultiModel train not implemented yet")
        for model in self.config.callables:
            model.train()

    def eval(self):
        for model in self.config.callables:
            model.eval()

    def __call__(self, *args, **kwds):
        return self.interface(*args, **kwds)
