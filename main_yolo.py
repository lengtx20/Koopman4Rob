"""Train data from Manipulation with Vision dataset using Deep Koopman model"""

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from models.deep_koopman import DeepKoopman
from runner.koopman_runner import KoopmanRunner
import os


def load_data(mode, data_dir, ratio: float = 0.8):
    traj = []
    for i in range(64):
        file_name = f"{i}.npy"
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path)
        traj.append(data)

    if mode == "test":
        train_traj = traj[: int(len(traj) * ratio)]
        val_traj = traj[int(len(traj) * ratio) :]
        train_traj = train_traj[:2]
        train_data = np.concatenate(train_traj, axis=0)
        val_data = np.concatenate(val_traj, axis=0)
        print(f"[INFO] Train data shape: {train_data.shape}")
        print(f"[INFO] Val data shape: {val_data.shape}")
    elif mode == "train":
        train_traj = traj[: int(len(traj) * ratio)]
        val_traj = traj[int(len(traj) * ratio) :]
        train_data = np.concatenate(train_traj, axis=0)
        val_data = np.concatenate(val_traj, axis=0)
        print(f"[INFO] Train data shape: {train_data.shape}")
        print(f"[INFO] Val data shape: {val_data.shape}")
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")
    return train_data, val_data


def run(mode="test", data_dir=None, model_dir=None, fisher_path=None):
    """
    mode:       select between train / test
    data_path:  path to the npy file.
                The structure of the data need to be (num_sample, x_t + a_t + x_t1).
    model_dir:  path to the Deep Koopman model.
                The model will be save to (or load from) this dir when training (or testing).
    """
    assert data_dir is not None, "Invalid data path."
    assert model_dir is not None, "Model path must be specified."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading Data from {data_dir}")
    train_data, val_data = load_data(mode, data_dir, ratio=0.8)

    # ===== init model and alg ===== #
    loss_fn = MSELoss()
    model = DeepKoopman(
        state_dim=7,
        action_dim=512,
        hidden_sizes=[512] * 3,
        lifted_dim=256,
        seed=42,
    )
    model.to_device(device)
    # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
    ewc = None
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    runner = KoopmanRunner(
        model=model,
        ewc_model=ewc,
        state_dim=7,
        action_dim=512,
        train_data=train_data,
        val_data=val_data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        normalize=False,
        ewc_lambda=100.0,
        batch_size=64,
        num_workers=0,
    )

    # ===== start main function ===== #
    if mode == "train":
        os.makedirs(os.path.dirname(model_dir) or ".", exist_ok=True)
        runner.train(
            max_epochs=250,
            save_model=True,
            model_dir=model_dir,
            task_id=1,
            fisher_path=fisher_path,
            # threshold_mode="neural_ratio",
            threshold_mode=None,
            ewc_threshold=1.0,
        )
        print(f"[INFO] Model saved to {model_dir}")
    elif mode == "test":
        runner.test(
            dataset="train", model_dir=model_dir, save_results=False, rollout_steps=1
        )
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")


if __name__ == "__main__":
    run(
        mode="train",
        data_dir="data/yolo/s_v_s1",
        model_dir="logs/yolo",
        # fisher_path="/home/ltx/Koopman4Rob/logs/test/ewc_task1.pt",
        fisher_path=None,
    )
