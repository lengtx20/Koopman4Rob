"""Train data from resnet with Vision dataset using Deep Koopman model"""

import torch
import torch.optim as optim
from models.deep_koopman import Deep_Koopman
from runner.koopman_runner import KoopmanRunner
from config import Config


def run(config: Config):
    config.device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        if not config.device
        else config.device
    )
    device = torch.device(config.device)
    model_cfg = config.model
    # ===== init model and alg ===== #
    loss_fn = config.train.loss_fn
    if isinstance(loss_fn, str):
        loss_fn = getattr(torch.nn, loss_fn)()
    model = Deep_Koopman(
        state_dim=model_cfg.state_dim,
        action_dim=model_cfg.action_dim,
        hidden_sizes=model_cfg.hidden_sizes,
        lifted_dim=model_cfg.lifted_dim,
        seed=config.seed,
    )
    model.to_device(device)
    # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
    # TODO: configure this, ref. DP project
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    runner = KoopmanRunner(model, None, None, optimizer, loss_fn, device, False, config)

    # ===== start main function ===== #
    model_dir = config.checkpoint_path
    mode = config.mode
    print(f"[INFO] Mode: {mode}. Model dir: {model_dir}")
    if mode == "train":
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
        runner.train()
    elif mode == "test":
        runner.test("train")
    elif mode == "infer":
        runner.infer()
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")


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
