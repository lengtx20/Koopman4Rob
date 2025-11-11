"""This file provides the implementation of the training procedure"""

import numpy as np
import torch
import datetime
import json
import yaml
import time
import shutil
from tqdm import tqdm
from utils.iter_manager import IterationManager
from utils.fifo_save import FIFOSave
from utils.utils import process_mse_losses, set_seed, get_model_size
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from config import Config
from pprint import pprint
from collections import defaultdict, Counter
from itertools import count
from torch import optim
from typing import Optional, Callable, Any
from mcap_data_loader.utils.extra_itertools import first_recursive
from mcap_data_loader.utils.array_like import get_tensor_device_auto
from mcap_data_loader.basis.cfgable import dump_or_repr


class KoopmanDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KoopmanRunner:
    def __init__(self, config: Config, train_data, val_data):
        if config.seed is not None:
            set_seed(config.seed)

        config.device = get_tensor_device_auto(config.device)
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
        first_batch = {}
        if config.datasets:
            from data.mcap_data_utils import create_dataloaders

            self._data_loaders = create_dataloaders(config)
            first_batch = first_recursive(
                self._data_loaders.values(), 3 if self.mode == "infer" else 2
            )
            interactor = self.config.interactor
            if interactor is not None:
                interactor.add_config(self.config)
                interactor.add_first_batch(first_batch)
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

        with torch.device(self.device):
            self.config.model.add_first_batch(first_batch)
            ckpt_dir = None if self.mode == "train" else config.checkpoint_path
            model = self.config.model.load(ckpt_dir)
            if self.mode == "train":
                self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
            else:
                model.eval()
            # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
            # TODO: configure this, ref. DP project
            self.model = model

    def iter_loader(self, stage: str, manager: Optional[IterationManager] = None):
        start_time = time.monotonic()
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
                manager.update_val_epoch(avg_loss, time.monotonic() - start_time)
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
                    self.model.save(model_path)
                    print(f"[Runner] Model saved to {model_path}")
        # print(f"Iteration {stage} took {time.monotonic() - start_time:.2f} seconds")
        return avg_loss

    def train(self):
        """
        Called by 'train' mode.
        """
        config = self.config
        ckpt_dir = config.checkpoint_path
        save_model = ckpt_dir is not None
        train_cfg = config.train
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
            for epoch in epoch_bar:
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
                    if manager.is_loss_improved("val"):
                        tqdm.write(
                            f"[INFO] Epoch {epoch + 1}: Validation loss improved to {val_loss:.5f} rad, "
                            f"RMSE: {np.sqrt(val_loss) / np.pi * 180:.3f} deg; Current train RMSE: {np.sqrt(train_loss) / np.pi * 180:.3f} deg"
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
            epoch -= 1
        print(f"Stop reasons: {manager.reasons}")

        total_time = (time.monotonic() - start_time) / 60.0  # in minutes
        total_time_per_epoch = total_time / (epoch + 1)
        batch_size = config.data_loader.batch_size
        metrics = {
            "time_stamp": {
                "start": start_timestamp,
                "end": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            },
            "epochs": epoch + 1,
            "model_size_mb": get_model_size(self.model),
            "total_time_minutes": total_time,
            "total_time_minutes_per_epoch": total_time_per_epoch,
            "total_time_minutes_per_epoch_no_batch": total_time_per_epoch * batch_size,
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
            self.model.save(path=last_model_path)
            print(f"[Runner] Last model saved to {last_model_path}")
            fifo_saver = self.improve_dict.get("val_loss", None)
            if fifo_saver.last_item is not None:
                shutil.copytree(fifo_saver.last_item, ckpt_dir / "best")
                print(f"[Runner] Best model copied to {ckpt_dir / 'best'}")
            np.save(ckpt_dir / "losses.npy", np.array(self.losses))
            np.save(ckpt_dir / "vales.npy", np.array(self.vales))
            with open(ckpt_dir / "training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            with open(ckpt_dir / "training_config.yaml", "w") as f:
                yaml.dump(config.model_dump(mode="json", fallback=dump_or_repr), f)
            print(f"[Runner] Losses, vales and metrics saved to {ckpt_dir}")
        else:
            print("[INFO] No Koopman model saved")

        # ----- compute fisher and save fisher info -----
        print("[INFO] Computing Fisher Information after training...")
        if self.ewc_model is not None:
            train_tensor = torch.tensor(
                self.train_data, dtype=torch.float32, device=self.device
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
