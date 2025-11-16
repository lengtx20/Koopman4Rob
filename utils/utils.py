import numpy as np
import statistics
from scipy.interpolate import make_interp_spline
from typing import Optional


def smooth_curve(x, y, points=300):
    x = np.array(x)
    y = np.array(y)
    t = np.linspace(0, 1, len(x))
    t_smooth = np.linspace(0, 1, points)
    x_spline = make_interp_spline(t, x)(t_smooth)
    y_spline = make_interp_spline(t, y)(t_smooth)
    return x_spline, y_spline


def process_mse_losses(losses: list) -> dict:
    loss_dict = {
        "mse": losses,
        "rmse_deg": np.sqrt(np.array(losses)) * 180 / np.pi,
    }
    loss_stat = {}
    for key, value in loss_dict.items():
        loss_stat[key] = {
            "mean": statistics.mean(value),
            "std": statistics.stdev(value) if len(value) > 1 else 0.0,
        }
    return loss_stat


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model_size(model, unit="MB"):
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
