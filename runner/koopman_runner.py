"""This file provides the implementation of the training procedure"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import datetime
import json
import time
from tqdm import tqdm
from utils.utils import smooth_curve
from utils.iter_manager import IterationManager
from utils.fifo_save import FIFOSave
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from config import Config
from pprint import pprint
from collections import defaultdict, Counter
from itertools import count
import statistics
import shutil


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
        model,
        train_data,
        val_data,
        optimizer,
        loss_fn,
        device,
        normalize=False,
        config: Config = None,
    ):
        """
        model:      Deep Koopman model
        normalize:  Current normalization relies on pre-defined mean and std value.
                    This is to assure the consistency of the model.
        """
        # params
        self.mode = config.mode
        self.model = model.to(device)
        self.ewc_model = config.ewc_model
        self.state_dim = config.model.state_dim
        self.action_dim = config.model.action_dim
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.normalize = normalize
        self.ewc_lambda = config.ewc_lambda
        self.num_workers = config.num_workers
        self.tb_log_dir = config.tb_log_dir
        self.config = config

        self.losses = []
        self.vales = []

        batch_size = config.batch_size
        mode = config.mode
        num_workers = config.num_workers

        # dataset
        if config.mode != "infer":
            if config.data_dir:
                from data.mcap_data_utils import create_train_val_dataloader

                self.train_loader, self.val_loader = create_train_val_dataloader(config)
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
        action_end = self.state_dim + self.action_dim
        a_t = sample[self.model.state_dim : action_end]
        x_t1 = sample[action_end:]
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

    def train(self):
        """
        Called by 'train' mode.
        """
        config = self.config
        ckpt_dir = config.checkpoint_path
        save_model = ckpt_dir is not None
        best_val_loss = float("inf")
        train_cfg = config.train
        if train_cfg.threshold_mode is not None:
            self.fisher_dict = None
            self.load_fisher(fisher_path=train_cfg.fisher_path)
        improve_dict: dict[str, FIFOSave] = {
            key: FIFOSave(max_count)
            for key, max_count in zip(
                train_cfg.save_model.on_improve, train_cfg.save_model.maximum
            )
        }
        # tensorboard
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_log_dir = self.tb_log_dir / f"{timestamp}"
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        print(f"[INFO] TensorBoard logs will be saved to {self.tb_log_dir}")

        manager = IterationManager(config.train.iteration)

        def iter_loader(stage: str):
            start_time_ = time.monotonic()
            sample_num = 0
            total_loss = 0
            loader = {"train": self.train_loader, "val": self.val_loader}[stage]
            if loader is None:
                return None
            batch: torch.Tensor

            if stage == "train":
                self.model.train()
            else:
                self.model.eval()
                no_grad = torch.no_grad()
                no_grad.__enter__()
            for batch in loader:
                if stage == "train":
                    self.optimizer.zero_grad()

                batch = batch.to(self.device)
                x_t = batch[:, : self.state_dim]
                a_t = batch[:, self.state_dim : self.state_dim + self.action_dim]
                x_t1 = batch[:, -self.state_dim :]

                pred_x_t1 = self.model(x_t, a_t, False)
                loss = self.loss_fn(pred_x_t1, x_t1)

                if stage == "train":
                    if (
                        train_cfg.ewc_regularization
                        and self.ewc_model is not None
                        and self.ewc_model.fisher is not None
                    ):
                        loss += self.ewc_lambda * self.ewc_model.penalty(self.model)
                    loss.backward()

                total_loss += loss.item() * batch.size(0)
                sample_num += batch.size(0)

                if stage == "train":
                    self.optimizer.step()
                    if manager.update_train_iter(batch.size(0)):
                        break
            avg_loss = total_loss / sample_num
            if stage == "train":
                if not manager.reasons:
                    manager.update_train_epoch(avg_loss)
            else:
                manager.update_val_epoch(avg_loss, time.monotonic() - start_time_)
                no_grad.__exit__(None, None, None)
            if save_model and manager.is_loss_improved(stage):
                # print(f"[INFO] {stage} loss improved to {avg_loss:.5f}")
                loss_key = f"{stage}_loss"
                saved_fifo = improve_dict.get(loss_key, None)
                if saved_fifo is not None:
                    model_path = ckpt_dir / f"{loss_key}/{avg_loss:.5f}"
                    saved_fifo.append(model_path)
                    self.model.save(model_dir=model_path)
                    print(f"[Runner] Model saved to {model_path}")
            return avg_loss

        epoch_bar = tqdm(
            range(config.train.iteration.max_epoch), desc="[Training]", position=0
        )
        start_time = time.monotonic()
        start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshots = defaultdict(list)
        manager.start()
        snap_count = Counter()
        print(f"[Runner] Training started: {start_time}...")
        try:
            for epoch_i, epoch in enumerate(epoch_bar):
                train_loss = iter_loader("train")
                train_reasons = manager.reasons
                val_loss = iter_loader("val")
                val_reasons = manager.reasons
                if val_reasons != train_reasons:
                    print(f"validation also: {train_reasons=}, {val_reasons=}")

                self.losses.append(train_loss)
                self.vales.append(val_loss)

                # ----- TensorBoard -----
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar("Loss/Val", val_loss, epoch)
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f"LR/Group{i}", param_group["lr"], epoch)

                # ----- tqdm postfix -----
                postfix = {"TrainLoss": f"{train_loss:.5f}"}
                if val_loss is not None:
                    postfix["ValLoss"] = f"{val_loss:.5f}"
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        tqdm.write(
                            f"[INFO] Epoch {epoch + 1}: Validation loss improved to {val_loss:.5f} rad, "
                            f"RMSE: {np.sqrt(val_loss) / np.pi * 180:.3f} deg"
                        )
                epoch_bar.set_postfix(postfix)
                for index, snap_cfg in enumerate(train_cfg.snapshot):
                    assert snap_cfg.unit == "minute", (
                        "Only 'minute' unit is supported now."
                    )
                    cost = manager.get_time_cost()
                    threshold = snap_cfg.interval * (snap_count[index] + 1)
                    if cost > threshold:
                        # print(f"{cost=} > {threshold=}, taking snapshot...")
                        for snap_key in snap_cfg.keys:
                            snapshots[snap_key].append(manager.records[snap_key])
                        snap_count[index] += 1
                if manager.reasons:
                    break
        except KeyboardInterrupt:
            manager.reasons.add("KeyboardInterrupt")

        print(f"Stop reasons: {manager.reasons}")

        total_time = (time.monotonic() - start_time) / 60.0  # in minutes
        total_time_per_epoch = total_time / (epoch_i + 1)

        def get_model_size(model: torch.nn.Module, unit="MB"):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

            size_bytes = param_size + buffer_size
            if unit == "KB":
                return size_bytes / 1024
            elif unit == "MB":
                return size_bytes / (1024**2)
            elif unit == "GB":
                return size_bytes / (1024**3)
            else:
                return size_bytes

        metrics = {
            "time_stamp": {
                "start": start_timestamp,
                "end": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            },
            "model_size_mb": get_model_size(self.model),
            "total_time_minutes": total_time,
            "total_time_minutes_per_epoch": total_time_per_epoch,
            "total_time_minutes_per_epoch_no_batch": total_time_per_epoch
            * config.batch_size,
            "iteration": manager.records,
            "snapshots": {
                "period": {
                    str(cfg.keys): f"{cfg.interval} {cfg.unit}"
                    for cfg in train_cfg.snapshot
                },
                "values": dict(snapshots),
            },
        }

        self.writer.close()

        # ----- save the last models, losses & vales and create a symbol link for the best val_loss model -----
        if save_model:
            last_model_path = ckpt_dir / "last"
            last_model_path.mkdir(parents=True, exist_ok=True)
            self.model.save(model_dir=last_model_path)
            print(f"[Runner] Last model saved to {last_model_path}")
            fifo_saver = improve_dict.get("val_loss", None)
            if fifo_saver is not None:
                shutil.copytree(fifo_saver.last_item, ckpt_dir / "best")
                print(f"[Runner] Best model copied to {ckpt_dir / 'best'}")
            np.save(ckpt_dir / "losses.npy", np.array(self.losses))
            np.save(ckpt_dir / "vales.npy", np.array(self.vales))
            with open(ckpt_dir / "training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"[Runner] Losses, vales and metrics saved to {ckpt_dir}")
        else:
            print("[INFO] No Koopman model saved")

        # ----- compute fisher and save fisher info -----
        print("[INFO] Computing Fisher Information after training...")
        if self.ewc_model is not None:
            train_tensor = torch.tensor(self.train_data, dtype=torch.float32).to(
                self.device
            )
            fisher = self.ewc_model.compute_fisher(train_tensor, batch_size=64)
            self.ewc_model.fisher = fisher
            print("[INFO] Fisher matrix updated.")
            if save_model and ckpt_dir is not None:
                self.ewc_model.save(model_dir=ckpt_dir, task_id=train_cfg.task_id)
        else:
            print("[INFO] No EWC model attached or invalid.")

    def test(self, dataset="val"):
        """
        Test model on given dataset (train/val).
        Args:
            dataset: str, "train" or "val"
            config: Config, configuration object containing test settings
        """
        # ----- load model -----
        config = self.config
        model_dir = config.checkpoint_path
        test_config = config.test
        rollout_steps = test_config.rollout_steps
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
        total_array_sum = torch.zeros(self.state_dim, device=self.device)
        array_max = torch.zeros(self.state_dim, device=self.device)
        array_min = torch.zeros(self.state_dim, device=self.device) + 1e4
        with torch.no_grad():
            start_time = time.monotonic()
            for index, batch in enumerate(loader):
                batch = batch.to(self.device)
                x_t = batch[:, : self.state_dim]
                action_end = self.state_dim + self.action_dim
                a_t = batch[:, self.state_dim : action_end]
                x_t1 = batch[:, action_end:]

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
                actual_bs = batch.size(0)
                loss = self.loss_fn(pred_x_t1, x_t1)
                total_loss += loss.item() * actual_bs
                n_samples += actual_bs
                abs_error = torch.abs(pred_x_t1 - x_t1)
                array_sum = abs_error.sum(dim=0)
                array_max = torch.maximum(array_max, abs_error.max(dim=0).values)
                array_min = torch.minimum(array_min, abs_error.min(dim=0).values)
                total_array_sum += array_sum
                total_mae += torch.mean(abs_error).item() * actual_bs

                # ----- record the trajectory -----
                # to cpu, do not waste gpu memory
                traj.append(torch.cat([x_t, a_t, x_t1, pred_x_t1], dim=1).cpu().numpy())

        # ----- summary -----
        avg_mse = total_loss / n_samples
        avg_rmse = np.sqrt(avg_mse)
        avg_rmse_deg = avg_rmse / np.pi * 180
        avg_array_mae = (total_array_sum / n_samples).cpu().numpy()
        metrics_dict = {
            "avg_time_per_batch": (time.monotonic() - start_time) / (index + 1),
            "avg_mse": avg_mse,
            "avg_rmse": avg_rmse,
            "avg_rmse_deg": avg_rmse_deg,
            "avg_mae": total_mae / n_samples,
            "avg_array_mae_rad": avg_array_mae.tolist(),
            "avg_array_mae_deg": (avg_array_mae * 57.2958).tolist(),
            "max_array_mae_deg": (array_max.cpu().numpy() * 57.2958).tolist(),
            "min_array_mae_deg": (array_min.cpu().numpy() * 57.2958).tolist(),
        }
        self.traj_np = np.concatenate(traj, axis=0)
        print("[INFO] Trajectory shape:", self.traj_np.shape)
        pprint(metrics_dict)

        # ----- save the results -----
        if test_config.save_results:
            np.save(model_dir / f"{dataset}_traj.npy", self.traj_np)
            with open(model_dir / f"{dataset}_metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"[Test-{dataset}] Results saved to {model_dir}")

        # ----- visualization -----
        if test_config.show_plot:
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

    def infer(self):
        from airbot_data_collection.common.systems.basis import SystemMode
        from airbot_data_collection.common.environments.grouped import (
            GroupedEnvironment,
            GroupedEnvironmentConfig,
            GroupsSendActionConfig,
        )
        from airbot_data_collection.common.systems.grouped import (
            SystemSensorComponentGroupsConfig,
            AutoControlConfig,
        )
        from airbot_data_collection.common.devices.cameras.v4l2 import (
            V4L2Camera,
            V4L2CameraConfig,
        )

        # from airbot_ie.robots.airbot_play import AIRBOTPlay, AIRBOTPlayConfig
        from airbot_ie.robots.airbot_play_mock import AIRBOTPlay, AIRBOTPlayConfig

        action_keys = self.config.robot_action_keys
        data_dir = self.config.data_dir
        use_dataset = data_dir is not None
        from mcap_data_loader.utils.pytorch import torch_to_numpy_dtype_dict

        if use_dataset:
            from mcap_data_loader.datasets.mcap_dataset import (
                McapFlatBuffersEpisodeDataset,
                DataRearrangeConfig,
                RearrangeType,
                get_config_and_class_type,
                to_episodic_sequence,
                get_first_sample,
            )
            from mcap_data_loader.utils.av_coder import DecodeConfig

            config_cls, dataset_cls = get_config_and_class_type(data_dir)

            dataset = dataset_cls(
                config_cls(
                    data_root=data_dir,
                    keys=self.config.image_keys + action_keys,
                    strict=False,
                    media_configs=[
                        DecodeConfig(mismatch_tolerance=5, frame_format="rgb24")
                    ],
                    rearrange=DataRearrangeConfig(
                        dataset=RearrangeType.SORT_STEM_DIGITAL
                    )
                    if dataset_cls is McapFlatBuffersEpisodeDataset
                    else DataRearrangeConfig(),
                )
            )
            dataset.load()
            dataset = to_episodic_sequence(dataset)
            reset_action = get_first_sample(dataset)[action_keys[0]]["data"].tolist()
            # McapFlatBuffersEpisodeDataset(

            # )
        else:
            reset_action = []
        env = GroupedEnvironment(
            GroupedEnvironmentConfig(
                components=SystemSensorComponentGroupsConfig(
                    groups=["/"] * 2,
                    names=["follow", "env_camera"],
                    roles=["l", "o"],
                    instances=[
                        AIRBOTPlay(AIRBOTPlayConfig()),
                        V4L2Camera(V4L2CameraConfig(camera_index=None)),
                    ],
                ),
                reset_action=GroupsSendActionConfig(
                    groups=["/"] if reset_action else [],
                    action_values=[reset_action],
                    modes=[SystemMode.RESETTING],
                ),
                auto_control=AutoControlConfig(groups=[]),
            )
        )

        # TODO: allow change the env reset action
        if not env.configure():
            raise RuntimeError("Failed to configure the grouped system.")

        action = GroupsSendActionConfig(
            groups=["/"],
            action_values=[[]],
            modes=[SystemMode.SAMPLING],
        )
        # ----- load model -----
        self.model.load(model_dir=self.config.checkpoint_path)
        self.model.eval()
        model_dtype = next(self.model.parameters()).dtype

        print("[INFO] Model loaded for inference.")

        from data.blip2_feature_extractor import Blip2ImageFeatureExtractor

        infer_cfg = self.config.infer
        blip2_cfg = infer_cfg.extra_models["blip2-itm-vit-g"]
        prompt = blip2_cfg["prompt"]
        if not prompt:
            raise ValueError("BLIP2 prompt is empty.")
        if not blip2_cfg["path"].exists():
            raise FileNotFoundError(f"BLIP2 model not found at {blip2_cfg['path']}")
        extractor = Blip2ImageFeatureExtractor(model_path=blip2_cfg["path"])
        extractor.load_model()
        freq = infer_cfg.frequency
        dt = 1 / freq if freq != 0 else 0
        pred_x_t1 = torch.tensor(
            reset_action, dtype=model_dtype, device=self.device
        ).unsqueeze(0)

        if infer_cfg.show_image:
            import cv2
            import einops

        from torchvision.transforms import v2

        transforms = [v2.ToImage()]
        if infer_cfg.image_transform:
            print("[INFO] Using image data augmentation transforms.")
            transforms.extend(
                [
                    v2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5)),
                    v2.RandomAdjustSharpness(sharpness_factor=2, p=1),
                ]
            )
        transforms.append(v2.ToDtype(extractor.dtype))
        image_transform = v2.Compose(transforms)
        all_losses = []
        with torch.no_grad():
            try:
                for rollout in count():
                    losses = []
                    max_rollouts = infer_cfg.max_rollouts
                    if max_rollouts > 0 and rollout > max_rollouts:
                        break
                    if use_dataset:
                        ep_ds = dataset[rollout]
                        sample_iter = iter(ep_ds)
                        print(f"[INFO] Dataset {ep_ds} loaded.")
                        # path = (
                        #     Path(f"{ep_ds.config.data_root.parent}_blip2_features")
                        #     / ep_ds.config.data_root.name
                        # )
                        # print(f"[INFO] Preparing BLIP2 features at {path}...")
                        # blip_dataset = McapFlatBuffersSampleDataset(
                        #     McapFlatBuffersSampleDatasetConfig(
                        #         data_root=path, keys=self.config.img_features_keys
                        #     )
                        # )
                        # blip_dataset.load()
                        # blip_iter = iter(blip_dataset)
                    env.reset()
                    if input(f"Press Enter to start {rollout=}...") == "q":
                        break
                    try:
                        for step in count():
                            max_steps = infer_cfg.max_steps
                            if max_steps > 0 and step > max_steps:
                                break
                            if use_dataset and (
                                infer_cfg.obs_from_dataset
                                or infer_cfg.action_from_dataset
                            ):
                                ds_obs = next(sample_iter)
                            if infer_cfg.obs_from_dataset:
                                obs = ds_obs
                            else:
                                obs = env.output().observation
                            cur_x_t_np = np.concatenate(
                                [obs[key]["data"] for key in action_keys],
                                dtype=torch_to_numpy_dtype_dict[model_dtype],
                            )
                            cur_x_t = (
                                torch.from_numpy(cur_x_t_np)
                                .to(device=self.device)
                                .unsqueeze(0)
                            )
                            # print(f"{cur_x_t=}")
                            # print(f"{pred_x_t1=}")
                            loss = torch.norm(cur_x_t - pred_x_t1) / np.pi * 180
                            print(f"RMSE: {loss} deg")
                            losses.append(loss.item())
                            if infer_cfg.open_loop_predict:
                                x_t = pred_x_t1
                            else:
                                x_t = cur_x_t

                            # print(obs.keys())
                            # print("Processing features..")
                            features = {}
                            if False:
                                pass
                                # blip2_obs = next(blip_iter)
                                # for key in self.config.img_features_keys:
                                #     features[key] = (
                                #         torch.from_numpy(blip2_obs[key]["data"])
                                #         .to(device=self.device, dtype=model_dtype)
                                #         .unsqueeze(0)
                                #     )
                            else:
                                for key in self.config.image_keys:
                                    img = obs[key]["data"]
                                    # raw_feature = extractor.process_image(img, prompt)[
                                    #     "features_proj"
                                    # ]
                                    # img: torch.Tensor = image_transform(img)
                                    features[key] = extractor.process_image(
                                        img, prompt
                                    )["features_proj"]
                                    # print(
                                    #     torch.norm(features[key] - raw_feature)
                                    #     / torch.norm(raw_feature)
                                    # )
                                    if infer_cfg.show_image:
                                        if isinstance(img, torch.Tensor):
                                            np_img = einops.rearrange(
                                                img.cpu()
                                                .float()
                                                .numpy()
                                                .astype(np.uint8),
                                                "c h w -> h w c",
                                            )
                                        else:
                                            np_img = img
                                        cv2.imshow(key, np_img[:, :, ::-1].copy())
                            if infer_cfg.show_image:
                                cv2.waitKey(1)
                            # TODO: support multiple image features
                            a_t = features[key]
                            a_t = a_t.to(dtype=model_dtype)
                            # print("Predicting...")
                            pred_x_t1 = self.model(x_t, a_t, False)
                            if infer_cfg.action_from_dataset:
                                # TODO: make sure using data from dataset
                                action.action_values = [cur_x_t_np.tolist()]
                            else:
                                action.action_values = [
                                    pred_x_t1.squeeze(0).cpu().numpy().tolist()
                                ]
                            if infer_cfg.send_action and not env.input(action):
                                print("Failed to send action, resetting...")
                                break
                            if dt > 0:
                                time.sleep(dt)
                            elif dt == 0:
                                input("Step done. Press Enter to continue...")
                    except KeyboardInterrupt:
                        print(f"Rollout interrupted by user at {step=}, resetting...")
                    except StopIteration:
                        print("Rollout stopped since dataset reached end")
                    mean_loss = statistics.mean(losses)
                    mean_std = statistics.stdev(losses) if len(losses) > 1 else 0.0
                    print(
                        f"Rollout {rollout} loss mean: {mean_loss} std: {mean_std} deg"
                    )
                    all_losses.append(mean_loss)
                    if infer_cfg.show_image:
                        cv2.destroyAllWindows()
            except KeyboardInterrupt:
                print("Inference session ended by user.")
            except StopIteration:
                print("Inference session ended since dataset reached end.")
        if all_losses:
            overall_mean = statistics.mean(all_losses)
            overall_std = statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0
            print(
                f"Overall inference mean loss: {overall_mean}, std: {overall_std} deg, over {len(all_losses)} rollouts."
            )
        env.shutdown()
