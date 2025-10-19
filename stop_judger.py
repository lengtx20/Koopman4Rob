from config import TrainIterationConfig
from collections import Counter
from pprint import pprint
import numpy as np
import time


def degree2loss(degree: float, loss: str) -> float:
    rad = degree / 180 * np.pi
    if loss == "mse":
        return rad**2
    elif loss == "mae":
        return rad
    raise ValueError(f"{loss} not supportted")


# common config choices
# config = TrainIterationConfig(iter_max=8)
# config = TrainIterationConfig(patience=5)
# config = TrainIterationConfig(max_time=0.0001)
loss = "mse"
min_train_loss = degree2loss(0.05, loss)
min_val_loss = degree2loss(0.1, loss)
train_loss = min_train_loss / 2
val_loss = min_val_loss / 2

config = TrainIterationConfig(
    min_train_loss=degree2loss(0.05, loss),
    min_val_loss=degree2loss(0.1, loss),
    iter_max=1000,
)


flags = {}
flag_stamps = {}
iter_counts = Counter()
iter_mode = config.iter_mode
last_losses = np.array([np.inf, np.inf])

start_time = time.monotonic()
for epoch_i, epoch in enumerate(range(10)):
    for bs_i, batch in enumerate(range(5)):
        actual_batch_size = 32
        if iter_mode != "epoch":
            iter_counts["step"] += 1
            iter_counts["sample"] += actual_batch_size
            cnt = iter_counts[iter_mode]
            # TODO: implement the stopping logic
            raise NotImplementedError
    iter_counts["epoch"] = epoch_i + 1
    if iter_mode == "epoch":
        time_cost = time.monotonic() - start_time
        cnt = iter_counts[iter_mode]
        flags["iter_max"] = cnt >= config.iter_max > 0
        flags["iter_min"] = cnt >= config.iter_min > 0
        losses = np.array([train_loss, val_loss])
        iter_counts["patience"] = (
            0 if np.any(losses < last_losses) else (flags.get("patience", 0) + 1)
        )
        flags["patience"] = iter_counts["patience"] >= config.patience > 0
        flags["max_train_loss"] = train_loss >= config.max_train_loss > 0.0
        flags["min_train_loss"] = train_loss <= config.min_train_loss > 0.0
        flags["max_val_loss"] = val_loss >= config.max_val_loss > 0.0
        flags["min_val_loss"] = val_loss <= config.min_val_loss > 0.0
        flags["max_time"] = time_cost >= config.max_time > 0.0
        last_losses = np.minimum(losses, last_losses)

        for key, flag in flags.items():
            if flag and (key not in flag_stamps):
                flag_stamps[key] = time_cost / 60.0

        # Check stopping criteria
        conds = config.conditions
        reasons = set()
        for key in conds.sufficient:
            if flags[key]:
                reasons = {key}
                break
        else:
            for key in conds.necessary:
                if not flags[key]:
                    break
            else:
                reasons = conds.necessary
        if reasons:
            print(f"{reasons=}")
            break

epoch_cnt = epoch_i + 1
if reasons == {"iter_max"}:
    assert epoch_cnt == config.iter_max, f"{epoch_cnt} vs {config.iter_max}"
elif reasons == {"patience"}:
    assert epoch_cnt == config.patience, f"{epoch_cnt} vs {config.patience}"

pprint(flag_stamps)
print("Done.")
