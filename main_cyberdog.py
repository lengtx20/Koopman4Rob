""" Train data from CyberDog dataset using Deep Koopman model """

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from models.deep_koopman import Deep_Koopman
from runner.koopman_runner import KoopmanRunner
from data.load_pickle_data import load_pickle_data
from data.cyberdog.data_processor import CyberDogDataProcessor
import os

def load_data(mode, data_dir, ratio: float=0.8):
    traj = []
    min_length = 1258
    for i in range(1, 2):           # 4
        for j in range(1, 4):       # 4
            for k in range(1, 6):   # 6
                file_name = f'cyberdog_{i}{j}{k}.npy'
                file_path = os.path.join(data_dir, file_name)
                data = np.load(file_path)
                traj.append(data)
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
    model = Deep_Koopman(
        state_dim=48,
        action_dim=6,
        hidden_sizes=[256] * 3,
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
            max_epochs=3000,
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
        runner.test(model_dir=model_dir)
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")

if __name__ == "__main__":
    run(
        mode="train",
        data_dir="data/cyberdog/converted_data",
        model_dir="logs/test_1.0",
        # fisher_path="/home/ltx/Koopman4Rob/logs/test/ewc_task1.pt",
        fisher_path=None,
    )
