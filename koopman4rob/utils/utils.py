import numpy as np
from scipy.interpolate import make_interp_spline


def smooth_curve(x, y, points=300):
    x = np.array(x)
    y = np.array(y)
    t = np.linspace(0, 1, len(x))
    t_smooth = np.linspace(0, 1, points)
    x_spline = make_interp_spline(t, x)(t_smooth)
    y_spline = make_interp_spline(t, y)(t_smooth)
    return x_spline, y_spline
