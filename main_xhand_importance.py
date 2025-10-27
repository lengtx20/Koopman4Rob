"""Train data from XHAND dataset using Deep Koopman model"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from models.deep_koopman import Deep_Koopman
from runner.koopman_runner import KoopmanRunner

def load_data(mode, data_dir, ratio: float = 1.0):
    """
    Load and preprocess XHAND dataset.

    params:
        mode: "train" or "test"
        data_dir: directory containing npz files
        ratio: train_data / all_data
    output:
        train_data, val_data
    """
    traj = []
    for i in [1,2,3,4,5,6]:           # force levels
        for j in [1]:       # block type
            for k in [1,2,3]:   # repetition
                file_name = f"{i}{j}{k}.npz.npy"
                file_path = os.path.join(data_dir, file_name)
                if not os.path.exists(file_path):
                    print(f"[WARN] Missing file: {file_name}")
                    continue
                x_and_u = np.load(file_path)
                traj.append(x_and_u)
                print(f"[INFO] Loaded: {file_name}")

    if len(traj) == 0:
        raise RuntimeError(f"[ERROR] No valid data found in {data_dir}")

    data = np.concatenate(traj, axis=0)
    print(f"[INFO] Data shape: {data.shape}")

    length = data.shape[0]
    if mode == "test":
        train_data = data[1000:1010]
        val_data = data[9000:9010]
    elif mode == "train":
        train_end = int(length * ratio)
        train_data = data[:train_end, :]
        val_data = data[train_end:, :]
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")
    return train_data, val_data


def run(mode="train", data_dir=None, load_model_dir=None, save_model_dir=None, fisher_path=None,
        freeze_encoder=False, freeze_decoder=False, freeze_matrix=False):
    """
    params:
        mode: 'train' or 'test'
        data_dir: directory containing XHAND data (.npz)
        load_model_dir: path to load pre-trained Koopman model
        save_model_dir: path to save model after training
    """
    assert data_dir is not None, "Invalid data path."
    assert save_model_dir is not None, "Save model path must be specified."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading Data from {data_dir}")
    train_data, val_data = load_data(mode, data_dir, ratio=1.0)

    # ===== init model and algorithm ===== #
    loss_fn = MSELoss()
    model = Deep_Koopman(
        state_dim=25,               # XHAND has 25-dimensional state
        action_dim=8,               # 8-dimensional action
        hidden_sizes=[256] * 3,
        lifted_dim=128,
        seed=42,
        iden_decoder=True,
    )
    model.to_device(device)

    # Load or freeze components
    if load_model_dir is not None:
        model.load(model_dir=load_model_dir)
    if freeze_matrix:
        model.freeze_matrix()
    if freeze_encoder:
        model.freeze_encoder()
    if freeze_decoder:
        model.freeze_decoder()

    ewc = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    runner = KoopmanRunner(
        mode=mode,
        state_dim=25,
        action_dim=8,
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

    # ===== main training/testing logic ===== #
    if mode == "train":
        print("[INFO] Training !")
        os.makedirs(os.path.dirname(save_model_dir) or ".", exist_ok=True)
        runner.train(
            max_epochs=30000,
            save_model=True,
            load_model_dir=load_model_dir,
            save_model_dir=save_model_dir,
            tb_log_dir=save_model_dir,
            task_id=1,
            fisher_path=fisher_path,
            threshold_mode=None,
            ewc_threshold=1.0,
        )
        print(f"[INFO] Model saved to {save_model_dir}")
    elif mode == "test":
        print("[INFO] Testing !")
        runner.test(load_model_dir=load_model_dir)
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}. Choose only between 'train' and 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test Deep Koopman model on XHAND dataset")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_dir", type=str, default="data/xhand/converted_data")
    parser.add_argument("--load_model_dir", type=str, default=None)
    parser.add_argument("--save_model_dir", type=str, default="EXPERIMENTS/1_importance_encoder_matrix/xhand/1")
    parser.add_argument("--fisher_path", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")
    parser.add_argument("--freeze_matrix", action="store_true")

    args = parser.parse_args()

    run(
        mode=args.mode,
        data_dir=args.data_dir,
        load_model_dir=args.load_model_dir,
        save_model_dir=args.save_model_dir,
        fisher_path=args.fisher_path,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        freeze_matrix=args.freeze_matrix,
    )
