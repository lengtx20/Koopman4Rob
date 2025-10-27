"""Train data from resnet with Vision dataset using Deep Koopman model"""

import torch
import torch.optim as optim
from torch.nn import MSELoss
from koopman4rob.models.deep_koopman import DeepKoopman
from koopman4rob.runner.koopman_runner import KoopmanRunner
import os


def run(
    mode="test", data_dir=None, model_dir=None, fisher_path=None, blip2_model_dir=""
):
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

    state_dim = 7
    action_dim = 256

    # ===== init model and alg ===== #
    loss_fn = MSELoss()
    model = DeepKoopman(
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
        mode=mode,
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
        data_root=data_dir,
        model_path=blip2_model_dir,
    )

    # ===== start main function ===== #
    if mode == "train":
        if os.path.exists(model_dir):
            idx = 0
            while True:
                new_dir = model_dir + f"{idx}"
                if not os.path.exists(new_dir):
                    model_dir = new_dir
                    print(f"[INFO] Model dir exists. Change to {model_dir}")
                    break
                idx += 1
        os.makedirs(os.path.dirname(model_dir) or ".", exist_ok=True)

        runner.train(
            max_epochs=150,
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
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    name = "blip2"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="test", help="select between train / test"
    )
    parser.add_argument(
        "--data-dir", "-dd", type=str, help="data directory", required=True
    )
    parser.add_argument(
        "--model-dir",
        "-md",
        type=str,
        default=f"logs/{name}_150",
        help="model directory",
    )
    parser.add_argument(
        "--blip2-model-dir", "-bd", type=str, help="BLIP2 model directory", default=""
    )
    parser.add_argument(
        "--fisher-path",
        "-fp",
        type=str,
        default=None,
        help="fisher information matrix path",
    )
    args = parser.parse_args()

    run(
        mode=args.mode,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        fisher_path=args.fisher_path,
        blip2_model_dir=args.blip2_model_dir,
    )
