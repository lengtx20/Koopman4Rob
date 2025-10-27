"""This file provides a demo main scirpt for training with Deep Koopman and computing EWC weights"""

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from koopman4rob.models.deep_koopman import DeepKoopman
from koopman4rob.runner.koopman_runner import KoopmanRunner
from koopman4rob.data.load_pickle_data import load_pickle_data
import os


def load_data(mode, data_path, ratio: int):
    # data: np.ndarray = np.load(data_path, allow_pickle=True)
    data: np.ndarray = load_pickle_data(data_path)
    print(f"[INFO] Data shape: {data.shape}")
    length = data.shape[0]
    if mode == "test":
        train_data = data[1000:1010]
        val_data = data[9000:9010]
    elif mode == "train":
        train_end = int(length * ratio)
        train_data = data[:train_end]
        val_data = data[train_end:]
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")
    return train_data, val_data


def run(mode="test", data_path=None, model_dir=None, fisher_path=None):
    """
    mode:       select between train / test
    data_path:  path to the npy file.
                The structure of the data need to be (num_sample, x_t + a_t + x_t1).
    model_dir:  path to the Deep Koopman model.
                The model will be save to (or load from) this dir when training (or testing).
    """
    assert data_path is not None, "Invalid data path."
    assert model_dir is not None, "Model path must be specified."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading Data from {data_path}")

    train_data, val_data = load_data(mode, data_path, ratio=0.8)

    # ===== init model and alg ===== #
    loss_fn = MSELoss()
    model = DeepKoopman(
        state_dim=7,
        action_dim=512 * 2,
        hidden_sizes=[128] * 3,
        lifted_dim=128,
        seed=42,
    )
    model.to_device(device)
    # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
    ewc = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    runner = KoopmanRunner(
        model=model,
        ewc_model=ewc,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        normalize=False,
        ewc_lambda=100.0,
    )

    # ===== start main function ===== #
    if mode == "train":
        os.makedirs(os.path.dirname(model_dir) or ".", exist_ok=True)
        runner.train(
            max_epochs=100000,
            model_dir=model_dir,
            task_id=1,
            fisher_path=fisher_path,
            # threshold_mode="neural_ratio",
            threshold_mode=None,
            ewc_threshold=1.0,
        )
        print(f"[INFO] Model saved to {model_dir}")
    elif mode == "test":
        runner.test(model_dir=model_dir)
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")


if __name__ == "__main__":
    run(
        mode="train",
        data_path="data/open_drawer_20_demos_2025-07-09_21-48-22.pkl",
        model_dir="logs/test_1.0",
        # fisher_path="/home/ltx/Koopman4Rob/logs/test/ewc_task1.pt",
        fisher_path=None,
    )
