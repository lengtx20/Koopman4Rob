from config import TrainIterationConfig
from collections import Counter
from typing import Set
import numpy as np
import time


class IterationManager:
    def __init__(self, config: TrainIterationConfig):
        self._config = config
        self.flags = {}
        self.flag_stamps = {}
        self.iter_counts = Counter()
        self.min_losses = {"train": np.inf, "val": np.inf}
        self.reasons = set()
        self._val_time_cost = 0.0
        self.start()

    def start(self):
        self.start_time = time.monotonic()

    def _update_flag_time(self, time_cost: float = 0.0):
        time_cost = self._get_time_cost() if time_cost == 0.0 else time_cost
        for key, flag in self.flags.items():
            if flag and (key not in self.flag_stamps):
                self.flag_stamps[key] = time_cost

    def _check_reasons(self, time_cost: float = 0.0) -> Set[str]:
        self._update_flag_time(time_cost)
        conds = self._config.conditions
        for key in conds.sufficient:
            if self.flags.get(key, False):
                self.reasons = {key}
                break
        else:
            for key in conds.necessary:
                if not self.flags.get(key, False):
                    break
            else:
                self.reasons = conds.necessary
        return self.reasons

    def _update_loss_flags(self, stage: str, loss: float):
        flags = self.flags
        new_min = loss < self.min_losses[stage]
        if new_min:
            min_loss_key = f"min_{stage}_loss"
            self.iter_counts["patience"] = 0
            self.min_losses[stage] = loss
            flags[min_loss_key] = loss <= getattr(self._config, min_loss_key) > 0.0
        else:
            self.iter_counts["patience"] += 1
            # print(f"Patience counter: {self.iter_counts['patience']}")
            # print(f"epoch {self.iter_counts['epoch']}")
            flags["patience"] = (
                self.iter_counts["patience"] >= self._config.patience > 0
            )
            max_loss_key = f"max_{stage}_loss"
            flags[max_loss_key] = loss >= getattr(self._config, max_loss_key) > 0.0

    def _get_time_cost(self) -> float:
        return (time.monotonic() - self.start_time - self._val_time_cost) / 60.0

    def _update_cnt_flags(self, keys: list[str]):
        for key in keys:
            cnt = self.iter_counts[key]
            self.flags[f"max_{key}"] = cnt >= getattr(self._config, f"max_{key}") > 0
            self.flags[f"min_{key}"] = cnt >= getattr(self._config, f"min_{key}") > 0

    def _update_iter_flags(self):
        self._update_cnt_flags(["step", "sample"])
        time_cost = self._get_time_cost()
        self.flags["max_time"] = time_cost >= self._config.max_time > 0.0
        self.flags["min_time"] = time_cost >= self._config.min_time > 0.0
        return time_cost

    def update_train_iter(self, batch_size: int) -> Set[str]:
        for key, delta in {"step": 1, "sample": batch_size}.items():
            self.iter_counts[key] += delta
        return self._check_reasons(self._update_iter_flags())

    def update_train_epoch(self, train_loss: float) -> Set[str]:
        self.iter_counts["epoch"] += 1
        self._update_loss_flags("train", train_loss)
        self._update_cnt_flags(["epoch"])
        return self._check_reasons()

    def update_val_epoch(self, val_loss: float, time_cost_sec: float = 0.0) -> Set[str]:
        self._val_time_cost += time_cost_sec
        self._update_loss_flags("val", val_loss)
        return self._check_reasons()
