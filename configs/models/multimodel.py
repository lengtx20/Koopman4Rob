from mcap_data_loader.callers.multi import MultiCaller, MultiCallerConfig
from mcap_data_loader.basis.cfgable import ConfigurableBasis
from basis import ModelLike
from typing import List, Union
from pathlib import Path
import yaml


class MultiModelConfig(MultiCallerConfig[ModelLike]):
    """Configuration for MultiModel"""


class MultiModel(ConfigurableBasis):
    interface: MultiCaller

    def __init__(self, config: MultiModelConfig):
        self.config = config
        self._models: List[ModelLike] = config.callables

    def on_configure(self):
        return self.interface.configure()

    def add_first_batch(self, batch) -> None:
        for model in self._models:
            model.add_first_batch(batch)
        if not self.configure():
            raise RuntimeError(
                "Failed to configure MultiModel after adding first batch"
            )

    def load(self, path: Union[List[Path], Path]):
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.suffix.removeprefix(".") in {"json", "yaml", "yml"}:
                path = yaml.safe_load(path.read_text())
                if isinstance(path, dict):
                    root = path.get("root", None)
                    paths = path["paths"]
                    suffixes = path.get("suffixes", "")

                    def kw_range(start, stop, step=1):
                        return range(start, stop, step)

                    if isinstance(paths, dict):
                        paths = kw_range(**paths)

                    if isinstance(suffixes, str):
                        suffixes = [suffixes] * len(paths)

                    path = (
                        paths
                        if root is None
                        else [Path(root) / str(p) / s for p, s in zip(paths, suffixes)]
                    )
                elif not isinstance(path, list):
                    raise ValueError("Expected a list of paths in the config file")
            else:
                path = [path]
        self.get_logger().info(f"Loading MultiModel from paths: {path}")
        if len(self._models) == 1:
            self.get_logger().info(
                "Only one model in MultiModel, replicating it for all paths"
            )
            self._models *= len(path)
        elif len(self._models) != len(path):
            raise ValueError(
                f"Number of models ({len(self._models)}) does not match "
                f"number of paths ({len(path)})"
            )
        for i, (model, p) in enumerate(zip(self._models, path)):
            self._models[i] = model.load(Path(p))
        return self

    def save(self, path: Path):
        for i, model in enumerate(self._models):
            model.save(path / str(i))

    def train(self):
        for model in self._models:
            model.train()

    def eval(self):
        for model in self._models:
            model.eval()

    def parameters(self):
        params = []
        for model in self._models:
            params.extend(model.parameters())
        return params

    def __call__(self, *args, **kwds):
        return self.interface(*args, **kwds)

    def __repr__(self):
        return str(list(self._models))
