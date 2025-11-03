import numpy as np
import statistics
from scipy.interpolate import make_interp_spline


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
