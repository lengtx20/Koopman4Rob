"""
Koopman Matrix Comparison Script
--------------------------------
This script loads Koopman matrices (G, A, K) from TWO different directories
and compares their properties side-by-side.

Metrics compared:
1. Spectral Radius (Stability)
2. Condition Number of G (Feature Redundancy)
3. Linearization Residual (How well K fits the data)
4. Orthogonality of Features

Usage:
    python analyze_compare.py --dir1 experiments/run1 --dir2 experiments/run2
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_matrices(matrix_dir):
    """Loads G, A, K matrices from the specified directory."""
    print(f"[INFO] Loading matrices from: {matrix_dir}")
    try:
        G = np.load(os.path.join(matrix_dir, "G_matrix.npy"))
        A = np.load(os.path.join(matrix_dir, "A_matrix.npy"))
        K = np.load(os.path.join(matrix_dir, "K_estimate.npy"))
        return G, A, K
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load files: {e}")
        return None, None, None


def compute_properties(G, A, K):
    """
    Computes scalar properties for the given matrices.
    Returns a dictionary of metrics.
    """
    props = {}

    # --- 1. Spectral Analysis (Eigenvalues of K) ---
    evals = np.linalg.eigvals(K)
    rho = np.max(np.abs(evals))
    props["spectral_radius"] = rho
    props["eigenvalues"] = evals  # Save for plotting

    # --- 2. G Matrix Analysis (Feature Quality) ---
    # Condition Number
    cond_g = np.linalg.cond(G)
    props["condition_number"] = cond_g

    # Numerical Rank
    rank_g = np.linalg.matrix_rank(G)
    props["rank"] = rank_g
    props["dim"] = G.shape[0]

    # Orthogonality Ratio (Sum of Diag^2 / Sum of Total^2)
    # 1.0 means perfectly orthogonal features
    diag_energy = np.sum(np.diag(G) ** 2)
    total_energy = np.sum(G**2)
    props["orthogonality"] = diag_energy / total_energy

    # --- 3. Hypothesis Check (Linearization Residual) ---
    # Metric: ||GK - A|| / ||A||
    # Measures how well the linear operator K maps the current features to future features
    residual = np.linalg.norm(G @ K - A) / np.linalg.norm(A)
    props["residual"] = residual

    return props


def plot_spectral_comparison(props1, props2, name1, name2):
    """
    Plots the eigenvalues of both sets on the unit circle.
    """
    plt.figure(figsize=(8, 8))

    # Draw Unit Circle
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")

    # Plot Set 1
    ev1 = props1["eigenvalues"]
    plt.scatter(
        ev1.real,
        ev1.imag,
        c="blue",
        alpha=0.6,
        marker="o",
        s=40,
        label=f"{name1}\n(rho={props1['spectral_radius']:.4f})",
    )

    # Plot Set 2
    ev2 = props2["eigenvalues"]
    plt.scatter(
        ev2.real,
        ev2.imag,
        c="red",
        alpha=0.6,
        marker="x",
        s=60,
        label=f"{name2}\n(rho={props2['spectral_radius']:.4f})",
    )

    plt.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    plt.title("Koopman Spectrum Comparison")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.show()


def run_comparison(dir1, dir2):
    # Load Data
    G1, A1, K1 = load_matrices(dir1)
    G2, A2, K2 = load_matrices(dir2)

    if G1 is None or G2 is None:
        print("[ERROR] Failed to load one or both datasets. Exiting.")
        return

    # Compute Metrics
    p1 = compute_properties(G1, A1, K1)
    p2 = compute_properties(G2, A2, K2)

    # Extract Names for display
    name1 = os.path.basename(os.path.normpath(dir1))
    name2 = os.path.basename(os.path.normpath(dir2))

    # --- Print Comparison Table ---
    # Define table structure
    headers = ["Metric", f"Set 1 ({name1})", f"Set 2 ({name2})", "Diff (1-2)"]
    row_fmt = "{:<25} | {:<20} | {:<20} | {:<20}"

    print("\n" + "=" * 95)
    print("   MATRIX PROPERTY COMPARISON")
    print("=" * 95)
    print(row_fmt.format(*headers))
    print("-" * 95)

    # List of metrics to display (Label, Key, Format)
    metrics_list = [
        ("Spectral Radius (Max EV)", "spectral_radius", "{:.6f}"),
        ("Condition Number (G)", "condition_number", "{:.4e}"),
        ("Matrix Rank", "rank", "{}"),
        ("Orthogonality Ratio", "orthogonality", "{:.4f}"),
        ("Linearization Residual", "residual", "{:.6e}"),
    ]

    for label, key, fmt in metrics_list:
        val1 = p1[key]
        val2 = p2[key]
        diff = val1 - val2

        # Format strings
        s1 = fmt.format(val1)
        s2 = fmt.format(val2)

        # Handle Diff formatting (Rank is integer, others float)
        if key == "rank":
            sd = f"{int(diff)}"
        else:
            sd = fmt.format(diff)

        print(row_fmt.format(label, s1, s2, sd))

    print("=" * 95 + "\n")

    # --- Visualize ---
    print("[INFO] Generating Spectral Comparison Plot...")
    plot_spectral_comparison(p1, p2, name1, name2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Koopman matrices from two directories."
    )

    parser.add_argument(
        "--dir1", type=str, required=True, help="Path to the first matrix directory"
    )
    parser.add_argument(
        "--dir2", type=str, required=True, help="Path to the second matrix directory"
    )

    args = parser.parse_args()

    run_comparison(args.dir1, args.dir2)
