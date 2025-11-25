"""Training, validating, testing and inferring the model all in one"""

from runner.koopman_runner import KoopmanRunner
from config import Config
from hydra_zen import instantiate, store
from omegaconf import DictConfig
from mcap_data_loader.utils.basic import ForceSetAttr
import hydra
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def run(config: Config):
    runner = KoopmanRunner(config, None, None)
    stage = config.stage
    model_dir = config.checkpoint_path
    logger.info(f"{stage=}, {model_dir=}")
    if stage == "train":
        # automatically change model dir if exists
        if model_dir.exists():
            ids = [int(p.stem) for p in model_dir.parent.iterdir() if p.stem.isdigit()]
            next_id = max(ids) + 1 if ids else 0
            model_dir = model_dir.parent / f"{next_id}{model_dir.suffix}"
        model_dir.mkdir(parents=True)
        with ForceSetAttr(config):
            config.checkpoint_path = model_dir
    runner.run(stage)


def main():
    config_name = "class_config"
    store(Config, name=config_name)
    store.add_to_hydra_store()

    config_path = "configs"
    config_name = "config"
    config = None

    def convert(cfg):
        nonlocal config
        config = instantiate(cfg)

    hydra.main(config_path or None, config_name, None)(convert)()
    if config is not None:
        print(type(config))
        if isinstance(config, DictConfig):
            config = Config(**config)
        run(config)


if __name__ == "__main__":
    main()
