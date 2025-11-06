"""Training, validating, testing and inferring the model all in one"""

from runner.koopman_runner import KoopmanRunner
from config import Config


def run(config: Config):
    runner = KoopmanRunner(config, None, None)
    mode = config.mode
    model_dir = config.checkpoint_path
    print(f"[INFO] Mode: {mode}. Model dir: {model_dir}")
    if mode == "train":
        # automatically change model dir if exists
        if model_dir.exists():
            ids = [int(p.stem) for p in model_dir.parent.iterdir() if p.stem.isdigit()]
            next_id = max(ids) + 1 if ids else 0
            model_dir = model_dir.parent / f"{next_id}{model_dir.suffix}"
        model_dir.mkdir(parents=True)
        config.checkpoint_path = model_dir
    runner.run(mode)


if __name__ == "__main__":
    import hydra
    from hydra_zen import instantiate, store

    config_name = "class_config"
    store(Config, name=config_name)
    store.add_to_hydra_store()

    # logging.basicConfig(level=logging.INFO)
    config_path = "configs"
    config_name = "config"
    config = None

    def convert(cfg):
        global config
        config = instantiate(cfg)

    hydra.main(config_path or None, config_name, None)(convert)()
    if config:
        run(Config(**config))
