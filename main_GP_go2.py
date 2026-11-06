"""
Script to compute Koopman Operators (G and P matrices) using a pre-trained Deep Koopman model.
"""

import os
import argparse
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from koopman4rob.models.deep_koopman import DeepKoopman

# from runner.koopman_analyzer_state import KoopmanDataset, KoopmanMathAnalyzer
from runner.koopman_analyzer_full import KoopmanDataset, KoopmanMathAnalyzer


def load_data(data_file):
    """
    Loads data from a single .pkl file.
    """
    print(f"[INFO] Loading data from {data_file}...")

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    # Handle if data is a list of trajectories (common in pkl) vs single array
    if isinstance(data, list):
        data = np.concatenate(data, axis=0)

    print(f"[INFO] Total Data shape: {data.shape}")
    return data


def run_analysis(
    data_file,
    load_model_dir,
    save_dir,
    batch_size=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data = load_data(data_file)
    dataset = KoopmanDataset(data)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("[INFO] Initializing Deep Koopman Model...")
    model = DeepKoopman(
        state_dim=48,
        action_dim=12,
        hidden_sizes=[128] * 3,
        lifted_dim=128,
        iden_decoder=True,
        device=device,
    )
    model.to_device(device)
    model.load(model_dir=load_model_dir)

    analyzer = KoopmanMathAnalyzer(model, device=device)

    print("\n" + "=" * 40)
    print("   Starting Formal Matrix Computation")
    print("=" * 40)

    G, P, K_opt = analyzer.compute_GP_matrices(data_loader)

    print(f"[RESULT] Computed G matrix shape: {G.shape}")
    print(f"[RESULT] Computed P matrix shape: {P.shape}")
    print(f"[RESULT] Computed K (estimate) shape: {K_opt.shape}")

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "G_matrix.npy"), G)
    np.save(os.path.join(save_dir, "A_matrix.npy"), P)
    np.save(os.path.join(save_dir, "K_estimate.npy"), K_opt)

    print(f"\n[INFO] Matrices saved to: {save_dir}")

    # Optional: Quick check on Condition Number of G
    cond_g = np.linalg.cond(G)
    print(f"[INFO] Condition number of G: {cond_g:.4e}")
    if cond_g > 1e10:
        print("[WARN] G is ill-conditioned. The inversion might be unstable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Koopman G and P matrices")

    # Changed to data_file
    parser.add_argument(
        "--data_file",
        type=str,
        default="koopman4rob/data/go2/gym_data/plane_20251116_175525.pkl",
        help="Path to the single .pkl data file",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default="koopman4rob/data/go2/trained_koopman_models/go2_plane",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="EXPERIMENTS/0_GP_go2/plane",
        help="Directory to save the computed G and P matrices",
    )
    args = parser.parse_args()

    run_analysis(
        data_file=args.data_file,
        load_model_dir=args.load_model_dir,
        save_dir=args.save_dir,
    )
