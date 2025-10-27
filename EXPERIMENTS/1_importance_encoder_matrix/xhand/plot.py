#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# file path
file_paths = ["1.csv", "1_freeze_encoder.csv", "1_freeze_matrix.csv"]

# legend name
labels = ["Full Model", "Matrix Fine-tune", "Lifting Function Fine-tune"]


# ======== EMA Smooth ========
def ema_smooth(values, alpha=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        last = alpha * last + (1 - alpha) * v
        smoothed.append(last)
    return np.array(smoothed)


# ======== IJRR Style ========
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.2,
        "figure.figsize": (6.0, 4.0),
    }
)

colors = ["#1f77b4", "#2ca02c", "#d62728"]  # blue, green, red


# ======== Plot ========
def plot_tensorboard_csvs():
    plt.figure()

    for idx, (file, label) in enumerate(zip(file_paths, labels)):
        if not os.path.exists(file):
            print(f"file not exist: {file}")
            continue

        df = pd.read_csv(file)
        if not {"Step", "Value"}.issubset(df.columns):
            print(f"{file} lack of Step/Value, skipping.")
            continue

        steps = df["Step"].values
        values = df["Value"].values

        smoothed = ema_smooth(values, alpha=0.9)

        # plt.plot(steps, values, color=colors[idx], alpha=0.25, linewidth=1.0)

        plt.plot(steps, smoothed, color=colors[idx], label=label, linewidth=2.2)

    # ======== plot ========
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()

    out_name = "ijrr_plot.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"Fig saved: {out_name}")
    plt.show()


# ======== main ========
if __name__ == "__main__":
    plot_tensorboard_csvs()
