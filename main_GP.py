"""
Script to compute Koopman Operators (G and P matrices) using a pre-trained Deep Koopman model.
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from koopman4rob.models.deep_koopman import DeepKoopman

# from runner.koopman_analyzer_state import KoopmanDataset, KoopmanMathAnalyzer
from runner.koopman_analyzer_full import KoopmanDataset, KoopmanMathAnalyzer


def load_data(data_dir):
    """
    Loads CyberDog data from the specified directory.
    Concatenates trajectories into a single dataset.
    """
    traj = []
    min_length = 1257

    # Specific file pattern matching your original script
    print(f"[INFO] Scanning {data_dir} for data files...")
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            for k in [1, 2, 3, 4, 5]:
                file_name = f"cyberdog_{i}{j}{k}.npy"
                file_path = os.path.join(data_dir, file_name)
                file_data = np.load(file_path)
                # Truncate to ensure consistent length if necessary
                truncated_data = file_data[:min_length, :]
                traj.append(truncated_data)

    data = np.concatenate(traj, axis=0)
    print(f"[INFO] Total Data shape: {data.shape}")
    return data


def run_analysis(
    data_dir,
    load_model_dir,
    save_dir,
    batch_size=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data = load_data(data_dir)
    dataset = KoopmanDataset(data)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("[INFO] Initializing Deep Koopman Model...")
    model = DeepKoopman(
        state_dim=48,
        action_dim=6,
        hidden_sizes=[256] * 3,
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

    parser.add_argument(
        "--data_dir", type=str, default="koopman4rob/data/cyberdog/converted_data"
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default="EXPERIMENTS/1_importance_encoder_matrix/cyberdog/1",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="EXPERIMENTS/0_GP_compute",
        help="Directory to save the computed G and P matrices",
    )
    args = parser.parse_args()

    run_analysis(
        data_dir=args.data_dir,
        load_model_dir=args.load_model_dir,
        save_dir=args.save_dir,
    )
