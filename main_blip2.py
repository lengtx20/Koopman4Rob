"""Train data from resnet with Vision dataset using Deep Koopman model"""

import torch
import torch.optim as optim
from torch.nn import MSELoss
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 7
    action_dim = 256

    # ===== init model and alg ===== #
    loss_fn = MSELoss()
    model = Deep_Koopman(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=[512] * 3,
        lifted_dim=256,
        seed=42,
    )
    model.to_device(device)
    # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
    ewc = None
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    runner = KoopmanRunner(
        mode=config.mode,
        model=model,
        ewc_model=ewc,
        state_dim=state_dim,
        action_dim=action_dim,
        train_data=None,
        val_data=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        normalize=False,
        ewc_lambda=100.0,
        batch_size=64,
        num_workers=0,
        config=config,
    )

    # ===== start main function ===== #
    model_dir = config.model_dir
    mode = config.mode
    if mode == "train":
        if model_dir.exists():
            idx = 0
            while True:
                new_dir = model_dir.parent / f"{model_dir.name}{idx}"
                if not new_dir.exists():
                    model_dir = new_dir
                    print(f"[INFO] Model dir exists. Change to {model_dir}")
                    break
                idx += 1
        model_dir.mkdir(parents=True, exist_ok=True)
        runner.train(
            max_epochs=150,
            save_model=True,
            model_dir=str(model_dir),
            task_id=1,
            fisher_path=config.fisher_path,
            # threshold_mode="neural_ratio",
            threshold_mode=None,
            ewc_threshold=1.0,
        )
        print(f"[INFO] Model saved to {model_dir}")
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
