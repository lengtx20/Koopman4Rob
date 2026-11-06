"""
Koopman Matrix Analysis Script
------------------------------
This script loads the computed G and A matrices (from main_GA.py) and performs:
1. Spectral Analysis (Eigenvalues of K) - Checks for system stability.
2. Condition Number Analysis - Checks if lifting functions are redundant.
3. Visualization - Plots eigenvalues relative to the unit circle.
4. Matrix Structure - Visualizes G and A heatmaps.
5. Hypothesis Testing - Checks Optimality Gap and Linearization Residual.

Usage:
    python analyze.py --matrix_dir results/koopman_matrices --model_path checkpoints/model
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Attempt to import the model structure for comparison
from koopman4rob.models.deep_koopman import DeepKoopman


def analyze_spectral_properties(K):
    """
    Computes and plots eigenvalues of the Koopman operator K.
    Checks for stability (eigenvalues <= 1).
    """
    print("\n--- Spectral Analysis ---")

    # Compute Eigenvalues
    evals, evecs = np.linalg.eig(K)

    # Sort by magnitude (descending)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]

    # Compute Spectral Radius
    rho = np.max(np.abs(evals))
    print(f"Spectral Radius (max |lambda|): {rho:.6f}")

    if rho > 1.0001:
        print("[WARNING] System appears UNSTABLE (eigenvalues outside unit circle).")
    else:
        print("[INFO] System appears STABLE.")

    # Plot
    plt.figure(figsize=(8, 8))

    # Draw Unit Circle
    theta = np.linspace(0, 2 * np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Unit Circle")

    # Plot Eigenvalues
    plt.scatter(
        evals.real,
        evals.imag,
        c="red",
        alpha=0.7,
        edgecolors="k",
        s=60,
        label="Eigenvalues",
    )

    # Annotate top 3 dominant modes
    for i in range(min(3, len(evals))):
        plt.annotate(f"{i + 1}", (evals[i].real, evals[i].imag + 0.05), fontsize=10)

    plt.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    plt.title(f"Koopman Spectrum\nSpectral Radius: {rho:.4f}")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    return evals


def analyze_gram_matrix(G):
    """
    Analyzes the Gram matrix G = Sum(psi * psi.T).
    G tells us about the orthogonality and independence of the learned features.
    """
    print("\n--- Gram Matrix (Feature) Analysis ---")

    # 1. Condition Number
    # High condition number (> 1e10) implies features are collinear/redundant.
    cond_g = np.linalg.cond(G)
    print(f"Condition Number of G: {cond_g:.4e}")

    if cond_g > 1e12:
        print(
            "[WARNING] G is extremely ill-conditioned. Inversion was likely unstable."
        )
    elif cond_g > 1e8:
        print("[WARNING] G is ill-conditioned. Features might be redundant.")
    else:
        print("[INFO] G is well-conditioned.")

    # 2. Rank
    rank = np.linalg.matrix_rank(G)
    print(f"Numerical Rank of G: {rank} / {G.shape[0]}")

    # 3. Off-diagonal energy (Orthogonality check)
    # If features were perfectly orthogonal, G would be diagonal.
    diag_energy = np.sum(np.diag(G) ** 2)
    total_energy = np.sum(G**2)
    orthogonality_ratio = diag_energy / total_energy
    print(
        f"Orthogonality Ratio: {orthogonality_ratio:.4f} (1.0 = Perfectly Orthogonal)"
    )


def analyze_hypothesis_metrics(G, A, K_analytical, model_path=None):
    """
    Performs the hypothesis tests:
    1. Linearization Residual: Can ANY linear matrix fit these features? (Checks Encoder quality)
    2. Optimality Gap: Did Gradient Descent find the best matrix? (Checks Optimizer quality)
    """
    print("\n" + "=" * 40)
    print("   Hypothesis Testing: Encoder vs Matrix")
    print("=" * 40)

    # --- Test 1: Linearizability (Residual) ---
    # Metric: ||GK - A|| / ||A||
    # If G is the covariance of features, and A is cross-covariance,
    # then GK = A is the Normal Equation. The residual measures how well K fits.
    residual = np.linalg.norm(G @ K_analytical - A) / np.linalg.norm(A)
    print(f"\n[1] Linearization Residual: {residual:.6e}")

    if residual > 0.1:
        print(
            "    >> HIGH Residual. The features defined by the Encoder are NOT linearly evolvable."
        )
        print("    >> Conclusion: The ENCODER is the primary bottleneck.")
    else:
        print("    >> LOW Residual. The features fit a linear model well.")

    # --- Test 2: Optimality Gap (Trained vs Analytical) ---
    if model_path and DeepKoopman is not None:
        try:
            print(f"\n[2] Loading trained model from {model_path} for comparison...")
            # Load weights manually to avoid instantiating full model if args missing
            # Assuming standard DeepKoopman save structure
            if os.path.isdir(model_path):
                A_path = os.path.join(model_path, "A.pth")
                if os.path.exists(A_path):
                    A_trained_torch = torch.load(A_path, map_location="cpu")
                    # DeepKoopman stores 'A' such that z_next = z @ A_trained.T (standard linear layer)
                    # Or sometimes explicitly z_next = z @ A.T.
                    # K_analytical is computed as z_next = z @ K_analytical
                    # So we compare K_analytical vs A_trained.T (or A_trained)

                    K_trained = A_trained_torch.detach().numpy().T

                    # Compute Gap
                    gap = np.linalg.norm(K_trained - K_analytical) / np.linalg.norm(
                        K_analytical
                    )
                    print(
                        f"    Optimality Gap (||K_trained - K_opt|| / ||K_opt||): {gap:.6f}"
                    )

                    if gap < 0.05:
                        print(
                            "    >> GAP IS SMALL. The trained matrix is nearly optimal."
                        )
                        print(
                            "    >> Hypothesis Supported: The Encoder sets the ceiling; the Matrix just fits it."
                        )
                    else:
                        print("    >> GAP IS LARGE. The trained matrix is suboptimal.")
                        print(
                            "    >> Hypothesis Weakened: Better matrix training could improve performance."
                        )
                else:
                    print(f"    [WARN] A.pth not found in {model_path}")
        except Exception as e:
            print(f"    [ERROR] Failed to compare with trained model: {e}")
    else:
        print(
            "\n[2] Optimality Gap: Skipped (No model_path provided or DeepKoopman not imported)"
        )


def visualize_matrices(G, A, K):
    """Plots heatmaps of the matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot G (Log scale helps visualize structure if dynamic range is large)
    sns.heatmap(np.log1p(np.abs(G)), ax=axes[0], cmap="viridis", cbar=True)
    axes[0].set_title("Log(|G|) - Gram Matrix")

    # Plot A
    sns.heatmap(np.log1p(np.abs(A)), ax=axes[1], cmap="viridis", cbar=True)
    axes[1].set_title("Log(|A|) - Interaction Matrix")

    # Plot K
    sns.heatmap(K.real, ax=axes[2], cmap="RdBu_r", center=0, cbar=True)
    axes[2].set_title("Real(K) - Operator")

    plt.tight_layout()


def run_analysis(matrix_dir, model_path=None):
    print(f"Loading matrices from: {matrix_dir}")

    try:
        G = np.load(os.path.join(matrix_dir, "G_matrix.npy"))
        A = np.load(os.path.join(matrix_dir, "A_matrix.npy"))
        K = np.load(os.path.join(matrix_dir, "K_estimate.npy"))
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load files: {e}")
        print("Ensure you ran main_GA.py first.")
        return

    print(f"Loaded dimensions: G{G.shape}, A{A.shape}, K{K.shape}")

    # 1. Analyze G (Features)
    analyze_gram_matrix(G)

    # 2. Analyze K (Dynamics)
    analyze_spectral_properties(K)

    # 3. Hypothesis Checks (New Addition)
    analyze_hypothesis_metrics(G, A, K, model_path)

    # 4. Visualization
    visualize_matrices(G, A, K)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix_dir",
        type=str,
        default="EXPERIMENTS/0_GA_go2/plane",
        help="Directory containing G_matrix.npy, etc.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional: Path to trained model directory (containing A.pth) for comparison.",
    )
    args = parser.parse_args()

    run_analysis(args.matrix_dir, args.model_path)
