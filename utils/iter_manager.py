from config import TrainIterationConfig
from collections import Counter
from typing import Set, Dict, Union, Tuple
import time


class IterationManager:
    def __init__(self, config: TrainIterationConfig):
        self._config = config
        self.flag_stamps = {}
        self._flags = {}
        self._iter_counts = Counter()
        self.min_losses = {"train": float("inf"), "val": float("inf")}
        self.last_losses = {}
        self.reasons = set()
        self._val_time_cost = 0.0
        self._is_loss_improved = {}
        self.start()

    def start(self):
        self.start_time = time.monotonic()

    def _update_flag_time(self, time_costs: Tuple[float, float] = (0.0, 0.0)):
        time_cost = self.get_time_cost() if time_costs[0] == 0.0 else time_costs[0]
        train_time_cost = (
            self.get_train_time_cost() if time_costs[1] == 0.0 else time_costs[1]
        )
        for key, flag in self._flags.items():
            if flag and (key not in self.flag_stamps):
                self.flag_stamps[key] = [time_cost, train_time_cost]

    def _check_reasons(self, time_costs: Tuple[float, float] = (0.0, 0.0)) -> Set[str]:
        self._update_flag_time(time_costs)
        conds = self._config.conditions
        for key in conds.sufficient:
            if self._flags.get(key, False):
                self.reasons = {key}
                break
        else:
            for key in conds.necessary:
                if not self._flags.get(key, False):
                    break
            else:
                self.reasons = conds.necessary
        return self.reasons

    def _update_loss_flags(self, stage: str, loss: float):
        flags = self._flags
        self.last_losses[stage] = loss
        new_min = loss < self.min_losses[stage]
        if new_min:
            self._is_loss_improved[stage] = True
            min_loss_key = f"min_{stage}_loss"
            self._iter_counts["patience"] = 0
            self.min_losses[stage] = loss
            flags[min_loss_key] = loss <= getattr(self._config, min_loss_key) > 0.0
        else:
            self._is_loss_improved[stage] = False
            self._iter_counts["patience"] += 1
            # print(f"Patience counter: {self.iter_counts['patience']}")
            # print(f"epoch {self.iter_counts['epoch']}")
            flags["patience"] = (
                self._iter_counts["patience"] >= self._config.patience > 0
            )
            max_loss_key = f"max_{stage}_loss"
            flags[max_loss_key] = loss >= getattr(self._config, max_loss_key) > 0.0
        # print(f"patience counter: {self._iter_counts['patience']}")

    def get_time_cost(self) -> float:
        return (time.monotonic() - self.start_time) / 60.0

    def get_train_time_cost(self) -> float:
        return (time.monotonic() - self.start_time - self._val_time_cost) / 60.0

    def _update_cnt_flags(self, keys: list[str]):
        for key in keys:
            cnt = self._iter_counts[key]
            self._flags[f"max_{key}"] = cnt >= getattr(self._config, f"max_{key}") > 0
            self._flags[f"min_{key}"] = cnt >= getattr(self._config, f"min_{key}") > 0

    def _update_iter_flags(self) -> Tuple[float, float]:
        self._update_cnt_flags(["step", "sample"])
        time_costs = []
        for t_key in ["", "train_"]:
            max_time_key = f"max_{t_key}time"
            min_time_key = f"min_{t_key}time"
            time_cost = getattr(self, f"get_{t_key}time_cost")()
            self._flags[max_time_key] = (
                time_cost >= getattr(self._config, max_time_key) > 0.0
            )
            self._flags[min_time_key] = (
                time_cost >= getattr(self._config, min_time_key) > 0.0
            )
            time_costs.append(time_cost)
        return time_costs

    def update_train_iter(self, batch_size: int) -> Set[str]:
        for key, delta in {"step": 1, "sample": batch_size}.items():
            self._iter_counts[key] += delta
        return self._check_reasons(self._update_iter_flags())

    def update_train_epoch(self, train_loss: float) -> Set[str]:
        self._iter_counts["epoch"] += 1
        self._update_loss_flags("train", train_loss)
        self._update_cnt_flags(["epoch"])
        return self._check_reasons()

    def update_val_epoch(self, val_loss: float, time_cost_sec: float = 0.0) -> Set[str]:
        self._val_time_cost += time_cost_sec
        self._update_loss_flags("val", val_loss)
        return self._check_reasons()

    def is_loss_improved(self, stage: str, ignore_first: bool = True) -> bool:
        if ignore_first and self._iter_counts["epoch"] == 1:
            return False
        return self._is_loss_improved.get(stage, False)

    @property
    def records(self) -> Dict[str, Union[float, list]]:
        return {
            "flag_stamps": self.flag_stamps,
            "min_train_loss": self.min_losses["train"],
            "min_val_loss": self.min_losses["val"],
            "train_loss": self.last_losses["train"],
            "val_loss": self.last_losses["val"],
            "stop_reasons": list(self.reasons),
        }
