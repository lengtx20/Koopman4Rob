"""This file provides the implementation of the training procedure"""

import numpy as np
import torch
import datetime
import json
import yaml
import time
import shutil
import traceback
from tqdm import tqdm
from utils.iter_manager import IterationManager
from utils.fifo_save import FIFOSave
from utils.utils import process_mse_losses, set_seed, get_model_size
from torch.utils.tensorboard import SummaryWriter
from config import Config
from pprint import pprint
from collections import defaultdict, Counter
from itertools import count
from torch import optim
from typing import Optional, Callable, Any
from mcap_data_loader.utils.extra_itertools import first_recursive
from mcap_data_loader.utils.array_like import get_tensor_device_auto
from mcap_data_loader.utils.basic import create_sleeper
from mcap_data_loader.basis.cfgable import dump_or_repr
from logging import getLogger
from interactor import ReturnAction, YieldKey


class RunnerExit(Exception):
    """Custom exit for the runner."""


class KoopmanRunner:
    def __init__(self, config: Config, train_data, val_data):
        set_seed(config.seed)
        config.device = get_tensor_device_auto(config.device)
        self.device = torch.device(config.device)
        self.loss_fn: Callable[[Any, Any], torch.Tensor] = config.loss_fn
        self.mode = config.stage
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

        # create data loaders and get the first batch
        first_batch = {}
        if config.datasets:
            from data.mcap_data_utils import create_dataloaders

            self._data_loaders = create_dataloaders(config)
            first_batch = first_recursive(
                self._data_loaders.values(), 3 if self.mode == "infer" else 2
            )
        # configure the interactor
        interactor = config.interactor
        if interactor is not None:
            interactor.add_config(config)
            """Since the interactor typically needs to be configured separately 
            for different episodes, such as with different initial conditions, 
            the `add_first_batch` method is not called here."""
        # load and prepare the model
        with torch.device(self.device):
            # Assume the model behaves consistently across all episodes.
            config.model.add_first_batch(first_batch)
            ckpt_dir = None if self.mode == "train" else config.checkpoint_path
            model = config.model.load(ckpt_dir)
            if self.mode == "train":
                # TODO: configure this, ref. DP project
                self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
            else:
                model.eval()
            # ewc = EWC(model, data=train_data, loss_fn=loss_fn, device=device)
            self.model = model

    def iter_loader(self, stage: str, manager: Optional[IterationManager] = None):
        start_time = time.monotonic()
        sample_num = 0
        total_loss = 0
        loader = self._data_loaders[stage]
        batch_losses = []

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
            elif stage == "test":
                batch_losses.append(loss.item())
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
                # self.get_logger().info(f"{stage} loss improved to {avg_loss:.5f}")
                loss_key = f"{stage}_loss"
                saved_fifo = self.improve_dict.get(loss_key, None)
                if saved_fifo is not None:
                    model_path = ckpt_dir / f"{loss_key}/{avg_loss:.5f}"
                    saved_fifo.append(model_path)
                    self.model.save(model_path)
                    self.get_logger().info(f"Model saved to {model_path}")
            return avg_loss
        # self.get_logger().info(f"Iteration {stage} took {time.monotonic() - start_time:.2f} seconds")
        return avg_loss, batch_losses

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
        logger = self.get_logger()
        logger.info(f"TensorBoard logs will be saved to {self.tb_log_dir}")
        manager = IterationManager(config.train.iteration)
        epoch_bar = tqdm(
            range(config.train.iteration.max_epoch), desc="[Training]", position=0
        )
        start_time = time.monotonic()
        start_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        snapshots = defaultdict(list)
        snap_count = Counter()
        logger.info(f"Training started: {start_time}...")
        manager.start()
        try:
            for epoch in epoch_bar:
                train_loss = self.iter_loader("train", manager)
                train_reasons = manager.reasons
                val_loss = self.iter_loader("val", manager)
                val_reasons = manager.reasons
                if val_reasons != train_reasons:
                    self.get_logger().info(
                        f"validation also: {train_reasons=}, {val_reasons=}"
                    )

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
                            f"Epoch {epoch + 1}: Validation loss improved to {val_loss:.5f} rad, "
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
                        # self.get_logger().info(f"{cost=} > {threshold=}, taking snapshot...")
                        for snap_key in snap_cfg.keys:
                            snapshots[snap_key].append(manager.records[snap_key])
                        snap_count[index] += 1
                if manager.reasons:
                    break
        except KeyboardInterrupt:
            manager.reasons.add("KeyboardInterrupt")
            epoch -= 1
        logger.info(f"Stop reasons: {manager.reasons}")

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
            logger.info(f"Last model saved to {last_model_path}")
            fifo_saver = self.improve_dict.get("val_loss", None)
            if fifo_saver.last_item is not None:
                shutil.copytree(fifo_saver.last_item, ckpt_dir / "best")
                logger.info(f"Best model copied to {ckpt_dir / 'best'}")
            np.save(ckpt_dir / "losses.npy", np.array(self.losses))
            np.save(ckpt_dir / "vales.npy", np.array(self.vales))
            with open(ckpt_dir / "training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
            with open(ckpt_dir / "training_config.yaml", "w") as f:
                yaml.dump(config.model_dump(mode="json", fallback=dump_or_repr), f)
            # to_yaml_file(
            #     ckpt_dir / "training_config.yaml",
            #     config,
            #     add_comments=False,
            #     fallback=dump_or_repr,
            # )
            logger.info(f"Losses, vales and metrics saved to {ckpt_dir}")
        else:
            logger.info("No Koopman model saved")

        # ----- compute fisher and save fisher info -----
        logger.info("Computing Fisher Information after training...")
        if self.ewc_model is not None:
            train_tensor = torch.tensor(
                self.train_data, dtype=torch.float32, device=self.device
            )
            fisher = self.ewc_model.compute_fisher(train_tensor, batch_size=64)
            self.ewc_model.fisher = fisher
            logger.info("Fisher matrix updated.")
            if save_model and ckpt_dir is not None:
                self.ewc_model.save(model_dir=ckpt_dir, task_id=train_cfg.task_id)
        else:
            logger.info("No EWC model attached or invalid.")

    def test(self):
        """Test model on given dataset (train/val)."""
        # ----- load model -----
        config = self.config
        test_cfg = config.test
        model_dir = config.checkpoint_path

        start_time = time.monotonic()
        # ----- computing loss -----
        avg_loss, batch_losses = self.iter_loader("test")
        # ----- summary -----
        avg_rloss = np.sqrt(avg_loss)
        avg_rloss_deg = avg_rloss / np.pi * 180
        metrics_dict = {
            "total_time": f"{(time.monotonic() - start_time):.2f} seconds",
            "avg_loss": avg_loss,
            "avg_rloss": avg_rloss,
            "avg_rloss_deg": avg_rloss_deg,
            "min_rloss_deg": np.sqrt(min(batch_losses)) / np.pi * 180,
            "max_rloss_deg": np.sqrt(max(batch_losses)) / np.pi * 180,
            "num_batches": len(batch_losses),
        }
        pprint(metrics_dict)

        # ----- save the results -----
        if test_cfg.save_results:
            with open(model_dir / f"{self.mode}_metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=4)
            self.get_logger().info(f"[Test-{self.mode}] Results saved to {model_dir}")

    def infer(self):
        data_loader = self._data_loaders.get(self.mode, None)
        use_data_loader = False
        if data_loader:
            use_data_loader = True
            num_loaders = len(data_loader)
        config = self.config
        infer_cfg = config.infer
        interactor = config.interactor
        freq = infer_cfg.frequency
        prediction = None
        rate = create_sleeper(freq)
        all_losses = []
        logger = self.get_logger()
        logger.info("Starting inference...")
        with torch.no_grad():
            try:
                for rollout in count():
                    losses = []
                    max_rollouts = infer_cfg.max_rollouts
                    if max_rollouts > 0 and rollout > max_rollouts:
                        break
                    if use_data_loader:
                        if config.data_loader.restart_on_stop_iteration:
                            index = rollout % num_loaders
                        else:
                            index = rollout
                        if index == 0 and rollout > 0:
                            logger.info("Iterating the data loader from the beginning")
                        cur_loader = data_loader[index]
                        first_batch = next(iter(cur_loader))
                        interactor.add_first_batch(first_batch)
                        ep_iter = iter(cur_loader)
                    if infer_cfg.rollout_wait is not None:
                        logger.info(
                            f"Press Enter to start {rollout=} or `q` to quit..."
                        )
                        if input() == "q":
                            break
                    rate.reset()
                    try:
                        for step in count():
                            max_steps = infer_cfg.max_steps
                            if max_steps > 0 and step > max_steps:
                                break
                            batch_data = next(ep_iter) if use_data_loader else {}
                            # self.get_logger().info("Predicting...")
                            # start = time.perf_counter()
                            generator = interactor.interact((prediction, batch_data))
                            try:
                                # get the model input
                                _, value = next(generator)[0]
                                prediction = self.model(value)
                                while True:
                                    # send back the prediction and get the next command
                                    yielded = generator.send((prediction, batch_data))
                                    if yielded is not None:
                                        for key, value in yielded:
                                            if key is YieldKey.NEXT_BATCH:
                                                batch_data = next(ep_iter)
                                            elif key is YieldKey.PREDICT:
                                                prediction = self.model(value)
                                            else:
                                                raise ValueError(
                                                    f"Unknown YieldKey: {key}"
                                                )
                                    # self.get_logger().info(
                                    #     "Prediction step took "
                                    #     f"{(time.perf_counter() - start):.4f} seconds"
                                    # )
                                    # start = time.perf_counter()
                                    rate.sleep()
                            except StopIteration as e:
                                value = e.value
                                if value is not None:
                                    logger.info(f"Interactor updating stopped: {value}")
                                    if value is ReturnAction.NEXT:
                                        break
                                    elif value is ReturnAction.EXIT:
                                        raise RunnerExit("interactor request")
                                    elif value is ReturnAction.WAIT:
                                        self.get_logger().info(
                                            "Press Enter to continue..."
                                        )
                                        input()
                                    elif isinstance(value, BaseException):
                                        raise value
                                    elif issubclass(value, BaseException):
                                        raise value("Interactor requested exception")
                                    else:
                                        pass
                            loss = self.loss_fn(prediction, batch_data)
                            losses.append(loss.item())
                            logger.info(f"RMSE: {torch.sqrt(loss) * 180 / np.pi} deg")
                    except KeyboardInterrupt:
                        logger.info(
                            f"Rollout interrupted by user at {step=}, resetting..."
                        )
                        if infer_cfg.rollout_wait is None:
                            logger.info("Press Enter to continue...")
                            input()
                    except StopIteration as e:
                        e_str = str(e)
                        if (e_str, e.__context__, e.__cause__) == ("", None, None):
                            logger.info(
                                f"Rollout {rollout} ended since dataset reached end."
                            )
                        elif e_str:
                            logger.info(f"Rollout stopped since: {e}")
                        else:
                            logger.info(
                                f"Rollout stopped due to an exception:\n{traceback.format_exc()}"
                            )
                    if losses:
                        loss_stats = process_mse_losses(losses)
                        loss_stats["rollout"] = rollout
                        pprint(loss_stats)
                        all_losses.append(loss_stats["mse"]["mean"])
            except KeyboardInterrupt:
                logger.info("Inference ended by user.")
            except (StopIteration, IndexError):
                logger.info("Inference ended since dataset reached end.")
            except RunnerExit as e:
                logger.info(f"Inference ended: {e}.")

        if all_losses:
            loss_stats = process_mse_losses(all_losses)
            loss_stats["rollout"] = "overall"
            logger.info(f"Overall loss stats: {loss_stats}")
        return interactor.shutdown()

    def run(self, stage: str):
        return getattr(self, stage)()

    @classmethod
    def get_logger(cls):
        return getLogger(cls.__name__)
