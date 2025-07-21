"""This file provides a demo main scirpt for training with Deep Koopman and computing EWC weights"""

import torch
import os
import numpy as np
import torch.optim as optim
from torch.nn import MSELoss
from models.deep_koopman import Deep_Koopman
from runner.koopman_runner import KoopmanRunner
from data.load_pickle_data import load_pickle_data
from airbot_data_collection.common.datasets.mcap_dataset import (
    McapFlatbufferEpisodeDataset,
    McapFlatbufferEpisodeDatasetConfig,
    DataSlicesConfig,
    DataRearrangeConfig,
)
from pathlib import Path


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


def run(data_root: str, mode="train", model_dir=None, fisher_path=None):
    """
    mode:       select between train / test
    data_root:  path to the data root.
    model_dir:  path to the Deep Koopman model.
                The model will be save to (or load from) this dir when training (or testing).
    """
    assert data_root is not None, "Invalid data path."
    assert model_dir is not None, "Model path must be specified."

    print(f"[INFO] Loading Data from {data_root}")
    number = len(list(Path(data_root).glob("*.mcap")))
    train_end = int(number * 0.8)
    print(f"[INFO] Found {number} mcap files in {data_root}")

    dataset_slices = {
        "train": (0, train_end),
        "val": (train_end, number),
    }
    datasets = {}
    for name, slices in dataset_slices.items():
        dataset = McapFlatbufferEpisodeDataset(
            McapFlatbufferEpisodeDatasetConfig(
                data_root=data_root,
                slices=DataSlicesConfig(dataset={data_root: slices}),
                rearrange=DataRearrangeConfig(
                    episode="sort",
                ),
                keys=[
                    "/lead/arm/joint_state/position",
                    "/lead/eef/joint_state/position",
                    "/env_camera/color/image_raw",
                    "/follow_camera/color/image_raw",
                ],
            )
        )
        dataset.load()
        datasets[name] = dataset

    # ===== init model and alg ===== #
    loss_fn = MSELoss()
    model = Deep_Koopman(
        state_dim=7,
        action_dim=512 * 2,
        hidden_sizes=[128] * 2,
        lifted_dim=128,
        seed=42,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to_device(device)
    # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
    ewc = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    runner = KoopmanRunner(
        model=model,
        ewc_model=ewc,
        train_data=datasets["train"],
        val_data=datasets["val"],
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
            max_epochs=10,
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
        data_root="/home/ghz/Work/OpenGHz/data-collection/airbot-data-collection/airbot_data_collection/data/red_meat_0",
        model_dir="logs/test_1.0",
        # fisher_path="/home/ltx/Koopman4Rob/logs/test/ewc_task1.pt",
        fisher_path=None,
    )
