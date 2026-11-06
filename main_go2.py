"""Train data from CyberDog dataset using Deep Koopman model"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from koopman4rob.models.deep_koopman import DeepKoopman
from koopman4rob.runner.koopman_runner import KoopmanRunner


def load_data(mode, data_dir, ratio: float = 0.9):
    """
    params:
        ratio: train_data / all_data
    output:
        train_data
        val_data
    """
    data = np.load(os.path.join(data_dir, "transitions_2025-11-12_15-42-19.npy"))
    print(f"[INFO] Data shape: {data.shape}")

    length = data.shape[0]
    if mode == "test":  # TODO
        train_data = data[1000:1010]
        val_data = data[9000:9010]
    elif mode == "train":
        train_end = int(length * ratio)
        train_data = data[:train_end, :]
        val_data = data[train_end:, :]
    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")
    return train_data, val_data


def run(
    mode="test",
    data_dir=None,
    load_model_dir=None,
    save_model_dir=None,
    fisher_path=None,
    freeze_encoder=False,
    freeze_decoder=False,
    freeze_matrix=False,
):
    """
    params:
        mode:       select between train / test
        data_dir:  path to the folder of npy file.
                    The structure of the data need to be (num_sample, x_t + a_t + x_t1).
        model_dir:  path to the Deep Koopman model.
                    The model will be save to (or load from) this dir when training (or testing).
    """
    assert data_dir is not None, "Invalid data path."
    assert load_model_dir is not None, "Load model path must be specified."
    assert save_model_dir is not None, "Save model path must be specified."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading Data from {data_dir}")
    train_data, val_data = load_data(mode, data_dir, ratio=0.8)

    # ===== init model and alg ===== #
    loss_fn = MSELoss()
    model = DeepKoopman(
        state_dim=48,
        action_dim=12,
        hidden_sizes=[256] * 3,
        lifted_dim=128,
        seed=42,
        iden_decoder=True,
    )
    model.to_device(device)

    # # explicit way
    # if load_model_dir is not None:
    #     model.load(model_dir=load_model_dir)
    # if freeze_matrix:
    #     model.freeze_matrix()
    # if freeze_encoder:
    #     model.freeze_encoder()
    # if freeze_decoder:
    #     model.freeze_decoder()

    ewc = None  # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    runner = KoopmanRunner(
        mode=mode,
        state_dim=48,
        action_dim=12,
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
        print("[INFO] Training !")
        os.makedirs(os.path.dirname(save_model_dir) or ".", exist_ok=True)
        runner.train(
            max_epochs=1000,
            save_model=True,
            load_model_dir=load_model_dir,
            save_model_dir=save_model_dir,
            tb_log_dir=save_model_dir,
            task_id=1,
            fisher_path=fisher_path,
            threshold_mode=None,  # threshold_mode="neural_ratio",
            ewc_threshold=1.0,
        )
        print(f"[INFO] Model saved to {save_model_dir}")
    elif mode == "test":
        print("[INFO] Testing !")
        runner.test(load_model_dir=load_model_dir)
    else:
        raise ValueError(
            f"[ERROR] Unknown mode: {mode}. Choose only between 'train' and 'test'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/test Deep Koopman model on Go2 Env in Simulation"
    )
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--data_dir", type=str, default="/home/ltx/Koopman4Rob/koopman4rob/data/go2"
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default="/home/ltx/Koopman4Rob/EXPERIMENTS/go2_model",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="/home/ltx/Koopman4Rob/EXPERIMENTS/go2_model",
    )
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
