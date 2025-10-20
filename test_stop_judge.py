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
# config = TrainIterationConfig(max_epoch=8)
# config = TrainIterationConfig(patience=5)
config = TrainIterationConfig(max_time=1.0 / 60)
loss = "mse"
min_train_loss = degree2loss(0.05, loss)
min_val_loss = degree2loss(0.1, loss)
train_loss = min_train_loss / 2
val_loss = min_val_loss / 2

# config = TrainIterationConfig(
#     min_train_loss=degree2loss(0.05, loss),
#     min_val_loss=degree2loss(0.1, loss),
#     iter_max=1000,
# )
manager = IterationManager(config)

start_time = time.monotonic()
epoch_i = -1
while True:
    epoch_i += 1
    for batch in range(5):
        actual_batch_size = 32
        if manager.update_train_iter(actual_batch_size):
            break
    else:
        if manager.update_train_epoch(train_loss):
            break
        val_start = time.monotonic()
        # do val here
        # ...
        # update
        if manager.update_val_epoch(val_loss, time.monotonic() - val_start):
            break
        continue
    break

epoch_cnt = epoch_i + 1
reasons = manager.reasons
if reasons:
    print(f"{reasons=}")
    if reasons == {"max_epoch"}:
        assert epoch_cnt == config.max_epoch, f"{epoch_cnt} vs {config.max_epoch}"
    elif reasons == {"patience"}:
        assert epoch_cnt + 1 == config.patience, f"{epoch_cnt} vs {config.patience}"
    elif reasons == {"max_time"}:
        cost_time = time.monotonic() - start_time
        assert 1.0 < cost_time < 1.2, f"{cost_time} vs {config.max_time}"
    else:
        raise ValueError(f"Unexpected reasons: {reasons}")
pprint(manager.flag_stamps)
print("Done.")
