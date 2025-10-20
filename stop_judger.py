from config import TrainIterationConfig
from pprint import pprint
from utils.iter_manager import IterationManager
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
manager = IterationManager(config)

start_time = time.monotonic()
for epoch_i, epoch in enumerate(range(10)):
    for bs_i, batch in enumerate(range(5)):
        actual_batch_size = 32
        if manager.update(train_loss, val_loss, actual_batch_size):
            break
    else:
        if manager.update(train_loss, val_loss):
            break
        continue
    break

epoch_cnt = epoch_i + 1
reasons = manager.reasons
if reasons:
    print(f"{reasons=}")
    if reasons == {"iter_max"}:
        assert epoch_cnt == config.iter_max, f"{epoch_cnt} vs {config.iter_max}"
    elif reasons == {"patience"}:
        assert epoch_cnt == config.patience, f"{epoch_cnt} vs {config.patience}"

pprint(manager.flag_stamps)
print("Done.")
