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
        self.min_losses = np.array([np.inf, np.inf])
        self.start_time = time.monotonic()
        self.reasons = set()

    def update(
        self, train_loss: float, val_loss: float, batch_size: int = 0
    ) -> Set[str]:
        config = self._config
        iter_counts = self.iter_counts
        delta = {"step": 1, "sample": batch_size}
        iter_mode = config.iter_mode
        if batch_size:
            if iter_mode != "epoch":
                iter_counts[iter_mode] += delta[iter_mode]
            else:
                return set()
        else:
            iter_counts["epoch"] += 1
        flags = self.flags
        time_cost = time.monotonic() - self.start_time
        cnt = iter_counts[config.iter_mode]
        flags["iter_max"] = cnt >= config.iter_max > 0
        flags["iter_min"] = cnt >= config.iter_min > 0
        losses = np.array([train_loss, val_loss])
        iter_counts["patience"] = (
            0 if np.any(losses < self.min_losses) else (flags.get("patience", 0) + 1)
        )
        flags["patience"] = iter_counts["patience"] >= config.patience > 0
        flags["max_train_loss"] = train_loss >= config.max_train_loss > 0.0
        flags["min_train_loss"] = train_loss <= config.min_train_loss > 0.0
        flags["max_val_loss"] = val_loss >= config.max_val_loss > 0.0
        flags["min_val_loss"] = val_loss <= config.min_val_loss > 0.0
        flags["max_time"] = time_cost >= config.max_time > 0.0
        self.min_losses = np.minimum(losses, self.min_losses)

        for key, flag in flags.items():
            if flag and (key not in self.flag_stamps):
                self.flag_stamps[key] = time_cost / 60.0

        # Check stopping criteria
        conds = config.conditions
        for key in conds.sufficient:
            if flags[key]:
                self.reasons = {key}
                break
        else:
            for key in conds.necessary:
                if not flags[key]:
                    break
            else:
                self.reasons = conds.necessary

        return self.reasons
