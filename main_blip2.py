"""Train data from resnet with Vision dataset using Deep Koopman model"""

from runner.koopman_runner import KoopmanRunner
from config import Config


def run(config: Config):
    runner = KoopmanRunner(config, None, None)

    # ===== start main function ===== #
    model_dir = config.checkpoint_path
    mode = config.mode
    print(f"[INFO] Mode: {mode}. Model dir: {model_dir}")
    if mode == "train":
        # automatically change model dir if exists
        if model_dir.exists():
            idx = 0
            while True:
                if model_dir.stem.isdigit():
                    stem = str(int(model_dir.stem) + idx)
                else:
                    stem = f"{model_dir.stem}{idx}"
                new_dir = model_dir.parent / f"{stem}{model_dir.suffix}"
                if not new_dir.exists():
                    model_dir = new_dir
                    print(f"[INFO] Model dir exists. Change to {model_dir}")
                    break
                idx += 1
        model_dir.mkdir(parents=True, exist_ok=True)
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
