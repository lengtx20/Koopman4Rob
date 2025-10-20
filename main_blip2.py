"""Train data from resnet with Vision dataset using Deep Koopman model"""

import torch
import torch.optim as optim
from models.deep_koopman import Deep_Koopman
from runner.koopman_runner import KoopmanRunner
from config import Config


def run(config: Config):
    """
    mode:       select between train / test
    data_path:  path to the npy file.
                The structure of the data need to be (num_sample, x_t + a_t + x_t1).
    model_dir:  path to the Deep Koopman model.
                The model will be save to (or load from) this dir when training (or testing).
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not config.device
        else torch.device(config.device)
    )
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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    runner = KoopmanRunner(model, None, None, optimizer, loss_fn, device, False, config)

    # ===== start main function ===== #
    model_dir = config.checkpoint_path
    mode = config.mode
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
        runner.train(config)
    elif mode == "test":
        runner.test("train", config)
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")


if __name__ == "__main__":
    import hydra
    from hydra_zen import instantiate, store

    config_name = "config"
    store(Config, name=config_name)
    store.add_to_hydra_store()

    # logging.basicConfig(level=logging.INFO)
    config_path = None
    config = None

    def convert(cfg):
        global config
        config = instantiate(cfg)

    hydra.main(config_path or None, config_name, None)(convert)()
    if config:
        run(config)
