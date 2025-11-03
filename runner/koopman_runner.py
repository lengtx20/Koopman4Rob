"""This file provides the implementation of the training procedure"""

import numpy as np
import torch
import datetime
import json
import time
import shutil
from tqdm import tqdm
from utils.iter_manager import IterationManager
from utils.fifo_save import FIFOSave
from utils.utils import process_mse_losses
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from config import Config
from pprint import pprint
from collections import defaultdict, Counter
from itertools import count
from models.deep_koopman import Deep_Koopman
from torch import optim
from typing import Optional, Callable, Any
from mcap_data_loader.utils.extra_itertools import first_recursive


class KoopmanDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KoopmanRunner:
    def __init__(self, config: Config, train_data, val_data):
        config.device = (
            config.device
            if config.device
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.device = torch.device(config.device)
        self.loss_fn: Callable[[Any, Any], torch.Tensor] = config.loss_fn
        self.mode = config.mode
        self.ewc_model = config.ewc_model
        self.train_data = train_data
        self.val_data = val_data
        self.normalize = config.data_loader.normalize
        self.ewc_lambda = config.ewc_lambda
        self.num_workers = config.data_loader.num_workers
        self.tb_log_dir = config.tb_log_dir
        self.config = config

        self.losses = []
        self.vales = []

        batch_size = config.data_loader.batch_size
        num_workers = self.num_workers

        # dataset
        if config.datasets:
            from data.mcap_data_utils import create_dataloaders

            self._data_loaders = create_dataloaders(config)
            interactor = self.config.interactor
            interactor.add_config(self.config)
            interactor.add_first_batch(
                first_recursive(
                    self._data_loaders.values(), 3 if self.mode == "infer" else 2
                )
            )

        elif config.mode != "infer":
            train_loader = DataLoader(
                KoopmanDataset(train_data),
                batch_size=batch_size,
                shuffle=(True and config.mode == "train"),
                num_workers=num_workers,
            )
            val_loader = (
                DataLoader(
                    KoopmanDataset(val_data),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
                if val_data is not None
                else None
            )
            self._data_loaders = {"train": train_loader, "val": val_loader}

        def create_model():
            return Deep_Koopman(
                **config.model.model_dump(),
                seed=config.seed,
            )

        if self.mode == "train":
            model = create_model()
        else:
            model_dir = config.checkpoint_path
            try:
                model = Deep_Koopman.load_from_checkpoint(model_dir=model_dir)
            except Exception as e:
                print(
                    "[Warning] Failed to load model from checkpoint using load_from_checkpoint:",
                    e,
                )
                model = create_model()
                model.load(model_dir=model_dir)
        model.to_device(self.device)
        if self.mode == "train":
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        else:
            model.eval()
        # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
        # TODO: configure this, ref. DP project
        self.model = model
        self.state_dim = config.model.state_dim
        self.action_dim = config.model.action_dim
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

    def iter_loader(self, stage: str, manager: Optional[IterationManager] = None):
        start_time_ = time.monotonic()
        sample_num = 0
        total_loss = 0
        loader = self._data_loaders[stage]
        if not loader:
            return None

        if stage == "train":
            self.model.train()
        else:
            self.model.eval()
            no_grad = torch.no_grad()
            no_grad.__enter__()
        for batch in loader:
            if stage == "train":
                self.optimizer.zero_grad()

            pred: torch.Tensor = self.model(batch)
            loss: torch.Tensor = self.loss_fn(pred, batch)

            if stage == "train":
                if (
                    self.config.train.ewc_regularization
                    and self.ewc_model is not None
                    and self.ewc_model.fisher is not None
                ):
                    loss += self.ewc_lambda * self.ewc_model.penalty(self.model)
                loss.backward()

            batch_size = batch["batch_size"]
            total_loss += loss.item() * batch_size
            sample_num += batch_size

            if stage == "train":
                self.optimizer.step()
                if manager.update_train_iter(batch_size):
                    break
        avg_loss = total_loss / sample_num
        if stage == "train":
            if not manager.reasons:
                manager.update_train_epoch(avg_loss)
        else:
            if stage == "val":
                manager.update_val_epoch(avg_loss, time.monotonic() - start_time_)
            no_grad.__exit__(None, None, None)
        if stage != "test":
            ckpt_dir = self.config.checkpoint_path
            save_model = ckpt_dir is not None
            if save_model and manager.is_loss_improved(stage):
                # print(f"[INFO] {stage} loss improved to {avg_loss:.5f}")
                loss_key = f"{stage}_loss"
                saved_fifo = self.improve_dict.get(loss_key, None)
                if saved_fifo is not None:
                    model_path = ckpt_dir / f"{loss_key}/{avg_loss:.5f}"
                    saved_fifo.append(model_path)
                    self.model.save(model_dir=model_path)
                    print(f"[Runner] Model saved to {model_path}")
        return avg_loss

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
            self.load_fisher(fisher_path=train_cfg.fisher_path)
        self.improve_dict: dict[str, FIFOSave] = {
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
        epoch_bar = tqdm(
            range(config.train.iteration.max_epoch), desc="[Training]", position=0
        )
        start_time = time.monotonic()
        start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshots = defaultdict(list)
        snap_count = Counter()
        print(f"[Runner] Training started: {start_time}...")
        manager.start()
        try:
            for epoch_i, epoch in enumerate(epoch_bar):
                train_loss = self.iter_loader("train", manager)
                train_reasons = manager.reasons
                val_loss = self.iter_loader("val", manager)
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
            * config.data_loader.batch_size,
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
            fifo_saver = self.improve_dict.get("val_loss", None)
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

    def test(self):
        """Test model on given dataset (train/val)."""
        # ----- load model -----
        config = self.config
        test_cfg = config.test
        model_dir = config.checkpoint_path

        start_time = time.monotonic()
        # ----- computing loss -----
        avg_loss = self.iter_loader("test")
        # ----- summary -----
        avg_rloss = np.sqrt(avg_loss)
        avg_rloss_deg = avg_rloss / np.pi * 180
        metrics_dict = {
            "total_time": f"{(time.monotonic() - start_time):.2f} seconds",
            "avg_loss": avg_loss,
            "avg_rloss": avg_rloss,
            "avg_rloss_deg": avg_rloss_deg,
        }
        pprint(metrics_dict)

        # ----- save the results -----
        if test_cfg.save_results:
            with open(model_dir / f"{self.mode}_metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"[Test-{self.mode}] Results saved to {model_dir}")

    def infer(self):
        data_loader = self._data_loaders.get(self.mode, None)
        use_data_loader = data_loader is not None
        infer_cfg = self.config.infer
        interactor = self.config.interactor

        freq = infer_cfg.frequency
        dt = 1 / freq if freq != 0 else 0
        prediction = None

        all_losses = []
        with torch.no_grad():
            try:
                for rollout in count():
                    losses = []
                    max_rollouts = infer_cfg.max_rollouts
                    if max_rollouts > 0 and rollout > max_rollouts:
                        break
                    if use_data_loader:
                        ep_iter = iter(data_loader[rollout])
                    if infer_cfg.rollout_wait is not None:
                        if (
                            input(f"Press Enter to start {rollout=} or `q` to quit...")
                            == "q"
                        ):
                            break
                    try:
                        for step in count():
                            max_steps = infer_cfg.max_steps
                            if max_steps > 0 and step > max_steps:
                                break
                            batch_data = next(ep_iter) if use_data_loader else {}
                            # print("Predicting...")
                            prediction = self.model(
                                interactor.get_model_input(prediction, batch_data)
                            )
                            interactor.update(prediction, batch_data)
                            loss = self.loss_fn(prediction, batch_data)
                            losses.append(loss.item())
                            print(f"RMSE: {torch.sqrt(loss) * 180 / np.pi} deg")
                            if dt > 0:
                                time.sleep(dt)
                            elif dt == 0:
                                input("Step done. Press Enter to continue...")
                    except KeyboardInterrupt:
                        print(f"Rollout interrupted by user at {step=}, resetting...")
                        if infer_cfg.rollout_wait is None:
                            input("Press Enter to continue...")
                    except StopIteration:
                        print("Rollout stopped since dataset reached end")
                    if losses:
                        loss_stats = process_mse_losses(losses)
                        loss_stats["rollout"] = rollout
                        pprint(loss_stats)
                        all_losses.append(loss_stats["mse"]["mean"])
            except KeyboardInterrupt:
                print("Inference session ended by user.")
            except (StopIteration, IndexError):
                print("Inference session ended since dataset reached end.")
        if all_losses:
            loss_stats = process_mse_losses(all_losses)
            loss_stats["rollout"] = "overall"
            pprint(loss_stats)
        return interactor.shutdown()

    def run(self, stage: str):
        return getattr(self, stage)()
