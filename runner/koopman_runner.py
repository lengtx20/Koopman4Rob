"""This file provides the implementation of the training procedure"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from utils.utils import smooth_curve
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from config import Config


class KoopmanDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KoopmanRunner:
    def __init__(
        self,
        mode,
        model,
        ewc_model,
        state_dim,
        action_dim,
        train_data,
        val_data,
        optimizer,
        loss_fn,
        device,
        normalize=False,
        batch_size=64,
        num_workers=0,
        ewc_lambda=0.0,
        tb_log_dir="logs/tensorboard",
        config: Config = None,
    ):
        """
        model:      Deep Koopman model
        normalize:  Current normalization relies on pre-defined mean and std value.
                    This is to assure the consistency of the model.
        """
        # params
        self.mode = mode
        self.model = model.to(device)
        self.ewc_model = ewc_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.normalize = normalize
        self.ewc_lambda = ewc_lambda
        self.num_workers = num_workers
        self.tb_log_dir = tb_log_dir

        self.losses = []
        self.vales = []

        # dataset
        if config.data_dir:
            from data.mcap_data_utils import create_train_val_dataloader

            self.train_loader, self.val_loader = create_train_val_dataloader(
                config,
                batch_size,
                num_workers,
                0.8 if mode == "train" else 1.0,
            )
        else:
            self.train_loader = DataLoader(
                KoopmanDataset(train_data),
                batch_size=batch_size,
                shuffle=(True and mode == "train"),
                num_workers=num_workers,
            )
            self.val_loader = (
                DataLoader(
                    KoopmanDataset(val_data),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
                if val_data is not None
                else None
            )

        # normalization (TODO)
        if self.normalize:
            self.state_mean = np.array(
                [-1.27054915, 0.94617132, -0.32996104, 5.84603260]
            )
            self.state_std = np.array([5.24317368, 4.13141372, 2.37722976, 2.74760289])
            self.action_mean = np.array([-1.14226600, -0.00369027])
            self.action_std = np.array([1.66356851, 0.32236316])
        else:
            self.mean = None
            self.std = None

    def _process_batch(self, data, i):
        """
        Split data into x_t, a_t, x_t1. Single sample at a time.
        If self.normalize = True, then sample retured will be normalized.
        """
        sample = data[i]
        x_t = sample[: self.model.state_dim]
        a_t = sample[
            self.model.state_dim : self.model.state_dim + self.model.action_dim
        ]
        x_t1 = sample[-self.model.state_dim :]
        if self.normalize:
            x_t = (x_t - self.state_mean) / self.state_std
            a_t = (a_t - self.action_mean) / self.action_std
            x_t1 = (x_t1 - self.state_mean) / self.state_std
        x_t = torch.tensor(x_t, dtype=torch.float32).to(self.device).unsqueeze(0)
        a_t = torch.tensor(a_t, dtype=torch.float32).to(self.device).unsqueeze(0)
        x_t1 = torch.tensor(x_t1, dtype=torch.float32).to(self.device).unsqueeze(0)
        return x_t, a_t, x_t1

    def _denormalize(self, data):
        if not self.normalize:
            return data
        state_dim = self.model.state_dim
        action_dim = self.model.action_dim

        x_t = data[:state_dim] * self.state_std + self.state_mean
        a_t = (
            data[state_dim : state_dim + action_dim] * self.action_std
            + self.action_mean
        )
        x_t1 = (
            data[state_dim + action_dim : state_dim * 2 + action_dim] * self.state_std
            + self.state_mean
        )
        pred_x_t1 = data[-state_dim:] * self.state_std + self.state_mean

        return np.concatenate((x_t, a_t, x_t1, pred_x_t1))

    def _evaluate_loss(self, data):
        """
        Compute the prediction loss by: loss = loss_fn (pred_x_t1, x_t1)
        """
        if data is None:
            return None

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(data.shape[0]):
                x_t, a_t, x_t1 = self._process_batch(data, i)
                pred_x_t1 = self.model(x_t, a_t, False)
                loss = self.loss_fn(pred_x_t1, x_t1)
                total_loss += loss.item()
        return total_loss / data.shape[0]

    def load_fisher(self, fisher_path, task_id=1):
        self.ckpt = torch.load(fisher_path)
        self.fisher_dict = self.ckpt.get("fisher_dict", {})
        print("[INFO] fisher_dict length:", len(self.fisher_dict))
        print("[INFO] fisher_dict keys:", self.fisher_dict.keys())
        if isinstance(self.fisher_dict, dict) and isinstance(
            list(self.fisher_dict.values())[0], dict
        ):
            self.fisher_dict = list(self.fisher_dict.values())[task_id - 1]

    def register_gradient_masks(self, threshold_mode, ewc_threshold):
        def create_mask(fisher_tensor):
            mask = torch.ones_like(fisher_tensor)
            if threshold_mode == "value":
                mask[fisher_tensor < ewc_threshold] = 0
            elif threshold_mode == "neural_ratio":
                thresh_val = torch.quantile(fisher_tensor.view(-1), ewc_threshold)
                mask[fisher_tensor < thresh_val] = 0
            elif threshold_mode == "weight_ratio":
                min_val = fisher_tensor.min()
                max_val = fisher_tensor.max()
                thresh_val = min_val + ewc_threshold * (max_val - min_val)
                mask[fisher_tensor < thresh_val] = 0
            else:
                raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
            return mask.to(self.device)

        # -------- param A --------
        param_A = self.model.A
        fisher_A = self.fisher_dict.get("A", None)
        if fisher_A is not None and fisher_A.shape == param_A.shape:
            mask_A = create_mask(fisher_A)
            param_A.register_hook(lambda grad: grad * mask_A)
        else:
            print("[Warning] No valid Fisher info for A")

        # -------- param B --------
        param_B = self.model.B
        fisher_B = self.fisher_dict.get("B", None)
        if fisher_B is not None and fisher_B.shape == param_B.shape:
            mask_B = create_mask(fisher_B)
            param_B.register_hook(lambda grad: grad * mask_B)
        else:
            print("[Warning] No valid Fisher info for B")

        # -------- encoder.layers.0.weight --------
        param_0_w = self.model.encoder.layers[0].weight
        fisher_0_w = self.fisher_dict.get("encoder.layers.0.weight", None)
        if fisher_0_w is not None and fisher_0_w.shape == param_0_w.shape:
            mask_0_w = create_mask(fisher_0_w)
            param_0_w.register_hook(lambda grad: grad * mask_0_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.0.weight")

        # -------- encoder.layers.0.bias --------
        param_0_b = self.model.encoder.layers[0].bias
        fisher_0_b = self.fisher_dict.get("encoder.layers.0.bias", None)
        if fisher_0_b is not None and fisher_0_b.shape == param_0_b.shape:
            mask_0_b = create_mask(fisher_0_b)
            param_0_b.register_hook(lambda grad: grad * mask_0_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.0.bias")

        # -------- encoder.layers.2.weight --------
        param_2_w = self.model.encoder.layers[2].weight
        fisher_2_w = self.fisher_dict.get("encoder.layers.2.weight", None)
        if fisher_2_w is not None and fisher_2_w.shape == param_2_w.shape:
            mask_2_w = create_mask(fisher_2_w)
            param_2_w.register_hook(lambda grad: grad * mask_2_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.2.weight")

        # -------- encoder.layers.2.bias --------
        param_2_b = self.model.encoder.layers[2].bias
        fisher_2_b = self.fisher_dict.get("encoder.layers.2.bias", None)
        if fisher_2_b is not None and fisher_2_b.shape == param_2_b.shape:
            mask_2_b = create_mask(fisher_2_b)
            param_2_b.register_hook(lambda grad: grad * mask_2_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.2.bias")

        # -------- encoder.layers.4.weight --------
        param_4_w = self.model.encoder.layers[4].weight
        fisher_4_w = self.fisher_dict.get("encoder.layers.4.weight", None)
        if fisher_4_w is not None and fisher_4_w.shape == param_4_w.shape:
            mask_4_w = create_mask(fisher_4_w)
            param_4_w.register_hook(lambda grad: grad * mask_4_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.4.weight")

        # -------- encoder.layers.4.bias --------
        param_4_b = self.model.encoder.layers[4].bias
        fisher_4_b = self.fisher_dict.get("encoder.layers.4.bias", None)
        if fisher_4_b is not None and fisher_4_b.shape == param_4_b.shape:
            mask_4_b = create_mask(fisher_4_b)
            param_4_b.register_hook(lambda grad: grad * mask_4_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.4.bias")

        # -------- encoder.layers.6.weight --------
        param_6_w = self.model.encoder.layers[6].weight
        fisher_6_w = self.fisher_dict.get("encoder.layers.6.weight", None)
        if fisher_6_w is not None and fisher_6_w.shape == param_6_w.shape:
            mask_6_w = create_mask(fisher_6_w)
            param_6_w.register_hook(lambda grad: grad * mask_6_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.6.weight")

        # -------- encoder.layers.6.bias --------
        param_6_b = self.model.encoder.layers[6].bias
        fisher_6_b = self.fisher_dict.get("encoder.layers.6.bias", None)
        if fisher_6_b is not None and fisher_6_b.shape == param_6_b.shape:
            mask_6_b = create_mask(fisher_6_b)
            param_6_b.register_hook(lambda grad: grad * mask_6_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.6.bias")

    def train(
        self,
        max_epochs=100,
        save_model=True,
        model_dir=None,
        task_id=1,
        ewc_regularization=False,
        fisher_path=None,
        threshold_mode=None,
        ewc_threshold=0.0,
    ):
        """
        Called by 'train' mode.
        """
        self.model.train()
        best_val_loss = float("inf")

        if threshold_mode is not None:
            self.fisher_dict = None
            self.load_fisher(fisher_path=fisher_path)

        # tensorboard
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_log_dir = os.path.join(self.tb_log_dir, f"{timestamp}")
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        print(f"[INFO] TensorBoard logs will be saved to {self.tb_log_dir}")

        epoch_bar = tqdm(range(max_epochs), desc="[Training]", position=0)
        for epoch in epoch_bar:
            # ----- training step -----
            total_loss = 0
            self.model.train()
            sample_num = 0
            for batch in self.train_loader:
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                x_t = batch[:, : self.state_dim]
                a_t = batch[:, self.state_dim : self.state_dim + self.action_dim]
                x_t1 = batch[:, -self.state_dim :]

                self.optimizer.zero_grad()
                pred_x_t1 = self.model(x_t, a_t, False)
                loss = self.loss_fn(pred_x_t1, x_t1)

                if ewc_regularization:
                    if self.ewc_model is not None and self.ewc_model.fisher is not None:
                        loss += self.ewc_lambda * self.ewc_model.penalty(self.model)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.size(0)
                sample_num += batch.size(0)

            train_loss = total_loss / sample_num

            # ----- validation step -----
            if self.val_loader is None:
                val_loss = None
            else:
                self.model.eval()
                val_loss = 0
                sample_num = 0
                with torch.no_grad():
                    for batch in self.val_loader:
                        batch = batch.to(self.device)
                        x_t = batch[:, : self.state_dim]
                        a_t = batch[
                            :, self.state_dim : self.state_dim + self.action_dim
                        ]
                        x_t1 = batch[:, -self.state_dim :]

                        pred_x_t1 = self.model(x_t, a_t, False)
                        loss = self.loss_fn(pred_x_t1, x_t1)
                        val_loss += loss.item() * batch.size(0)
                        sample_num += batch.size(0)

                val_loss /= sample_num

            self.losses.append(train_loss)
            self.vales.append(val_loss)

            # ----- TensorBoard -----
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f"LR/Group{i}", param_group["lr"], epoch)

            # ----- tqdm postfix -----
            postfix = {"TrainLoss": f"{train_loss:.4f}"}
            if val_loss is not None:
                postfix["ValLoss"] = f"{val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tqdm.write(
                        f"[INFO] Epoch {epoch + 1}: Validation loss improved to {val_loss:.4f}"
                    )
            epoch_bar.set_postfix(postfix)

        self.writer.close()

        # ----- save models and losses & vales -----
        if save_model and model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir=model_dir)
            print(f"[Runner] Model saved to {model_dir}")
        else:
            print("[INFO] No Koopman model saved")

        if save_model and model_dir is not None:
            self.model.save(model_dir=model_dir)
            print(f"[Runner] Model saved to {model_dir}")
        else:
            print("[INFO] No Koopman model saved")

        if save_model and model_dir is not None:
            np.save(os.path.join(model_dir, "losses.npy"), self.losses)
            np.save(os.path.join(model_dir, "vales.npy"), self.vales)
            print(f"[Runner] Losses and vales saved to {model_dir}")

        # ----- compute fisher and save fisher info -----
        print("[INFO] Computing Fisher Information after training...")
        if self.ewc_model is not None:
            train_tensor = torch.tensor(self.train_data, dtype=torch.float32).to(
                self.device
            )
            fisher = self.ewc_model.compute_fisher(train_tensor, batch_size=64)
            self.ewc_model.fisher = fisher
            print("[INFO] Fisher matrix updated.")
            if save_model and model_dir is not None:
                self.ewc_model.save(model_dir=model_dir, task_id=task_id)
        else:
            print("[INFO] No EWC model attached or invalid.")

    def test(
        self,
        dataset="val",
        model_dir=None,
        show_plot=True,
        save_results=True,
        rollout_steps=1,
    ):
        """
        Test model on given dataset (train/val).
        Args:
            dataset: str, "train" or "val"
            model_dir: str, path to load model
            show_plot: bool, whether to visualize trajectories
            save_results: bool, whether to save trajectory and metrics
            rollout_steps: int, number of steps for rollout prediction (>=1)
        """
        # ----- load model -----
        if model_dir is not None:
            self.model.load(model_dir=model_dir)
        self.model.eval()

        # ----- select dataset -----
        if dataset == "train":
            loader = self.train_loader
        elif dataset == "val":
            loader = self.val_loader
        else:
            raise ValueError("dataset must be 'train' or 'val'")

        traj = []
        total_loss, total_mae = 0, 0
        n_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                x_t = batch[:, : self.state_dim]
                a_t = batch[:, self.state_dim : self.state_dim + self.action_dim]
                x_t1 = batch[:, -self.state_dim :]

                # ----- 单步 or 多步预测 -----
                if rollout_steps == 1:
                    pred_x_t1 = self.model(x_t, a_t, False)
                else:
                    # rollout N 步，输入是当前 state，动作来自数据
                    pred_x_t1 = []
                    cur_x = x_t
                    for step in range(rollout_steps):
                        cur_a = batch[
                            :, self.state_dim : self.state_dim + self.action_dim
                        ]
                        cur_x = self.model(cur_x, cur_a, False)
                        pred_x_t1.append(cur_x)
                    pred_x_t1 = pred_x_t1[-1]  # 取最后一步预测

                # ----- loss & metrics -----
                loss = self.loss_fn(pred_x_t1, x_t1)
                total_loss += loss.item() * batch.size(0)
                total_mae += torch.mean(
                    torch.abs(pred_x_t1 - x_t1)
                ).item() * batch.size(0)
                n_samples += batch.size(0)

                # ----- 保存轨迹 -----
                traj.append(
                    torch.cat(
                        [x_t.cpu(), a_t.cpu(), x_t1.cpu(), pred_x_t1.cpu()], dim=1
                    ).numpy()
                )

        # ----- 结果统计 -----
        avg_loss = total_loss / n_samples
        avg_mae = total_mae / n_samples
        self.traj_np = np.concatenate(traj, axis=0)
        print("[INFO] Trajectory shape:", self.traj_np.shape)

        print(f"[Test-{dataset}] Avg Loss: {avg_loss:.4f}, Avg MAE: {avg_mae:.4f}")

        # ----- 保存结果 -----
        if save_results and model_dir is not None:
            np.save(os.path.join(model_dir, f"{dataset}_traj.npy"), self.traj_np)
            with open(os.path.join(model_dir, f"{dataset}_metrics.txt"), "w") as f:
                f.write(f"Avg Loss: {avg_loss}\nAvg MAE: {avg_mae}\n")
            print(f"[Test-{dataset}] Results saved to {model_dir}")

        # ----- 可视化 -----
        if show_plot:
            self.plot_trajectory()

    def plot_trajectory(self, use_smooth=True):
        """
        绘制每个状态变量随时间的轨迹（真实 vs 预测）。
        x 轴为时间步，y 轴为状态值。
        """
        x_t = self.traj_np[:, : self.model.state_dim]  # True states
        pred_x_t1 = self.traj_np[:, -self.model.state_dim :]  # Predicted states

        print("[DEBUG] x_t shape:", x_t.shape)
        print("[DEBUG] pred_x_t1 shape:", pred_x_t1.shape)
        time_idx = np.arange(len(x_t))
        num_dims = self.model.state_dim
        ncols = min(2, num_dims)  # 每行最多 2 个
        nrows = (num_dims + ncols - 1) // ncols

        plt.figure(figsize=(6 * ncols, 3 * nrows))

        for i in range(num_dims):
            ax = plt.subplot(nrows, ncols, i + 1)

            if use_smooth:
                t_s, y_s = smooth_curve(time_idx, x_t[:, i])
                t_p, y_p = smooth_curve(time_idx, pred_x_t1[:, i])
            else:
                t_s, y_s = time_idx, x_t[:, i]
                t_p, y_p = time_idx, pred_x_t1[:, i]

            ax.plot(t_s, y_s, label="True", color="blue")
            ax.plot(t_p, y_p, label="Pred", color="orange")
            ax.scatter(time_idx, x_t[:, i], color="blue", s=8, alpha=0.3)
            ax.scatter(time_idx, pred_x_t1[:, i], color="orange", s=8, alpha=0.3)

            ax.set_title(f"State[{i}] over Time")
            ax.set_xlabel("Time step")
            ax.set_ylabel(f"State {i}")
            ax.legend()

        plt.tight_layout()
        plt.show()
