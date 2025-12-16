"""Training, validating, testing and inferring the model all in one"""

from runner.koopman_runner import KoopmanRunner
from config import Config
from mcap_data_loader.configurers.basis import main_argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def run(config: Config):
    stage = config.stage
    ckpt_path = config.checkpoint_path
    logger.info(f"{stage=}, {ckpt_path=}")
    KoopmanRunner(config).run(stage)
    return 0


def main():
    configurer = main_argparse()(Config)
    configurer.parse("configs/config.yaml")
    return configurer.configure(run)


if __name__ == "__main__":
    main()
