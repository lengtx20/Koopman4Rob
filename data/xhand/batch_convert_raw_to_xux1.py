"""
This script converts (x_t, u_t) sequences into (x_t, u_t, x_{t+1}) format,
and saves the results as .npz files.

Batch processing: converts all .npz files in the given directory.

Args:
    input_dir (str): Directory containing input .npz files (each as [x_t, u_t]).
    output_dir (str): Directory to save converted .npz files.
    n (int): Dimension of the state vector x_t.
    m (int): Dimension of the control vector u_t.
"""

import numpy as np
import os
from data_loader import extract_flattened_data_for_xhand


def convert_xu_to_xux1(data: np.ndarray, n: int, m: int):
    """Convert (x_t, u_t) → (x_t, u_t, x_{t+1})."""
    assert data.shape[1] == n + m, (
        f"Expected dimension {n + m}, but got {data.shape[1]}"
    )
    x_u_t = data[:-1, :]  # (T-1, n+m)
    x_tp1 = data[1:, :n]  # (T-1, n)
    xu_x1 = np.concatenate([x_u_t, x_tp1], axis=1)  # (T-1, n+m+n)
    return xu_x1


def batch_process_npz_files(input_dir: str, output_dir: str, n: int, m: int):
    """Batch process all .npz files in a folder."""
    os.makedirs(output_dir, exist_ok=True)
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    if not npz_files:
        print(f"[WARN] No .npz files found in {input_dir}")
        return

    for filename in npz_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            data = extract_flattened_data_for_xhand(file_path=filename)
            converted = convert_xu_to_xux1(data, n, m)
            np.save(output_path, converted)
            print(f"[INFO] Converted: {filename}, shape: {converted.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to convert {filename}: {e}")


if __name__ == "__main__":
    input_folder = os.getcwd()  # current folder
    output_folder = os.path.join(input_folder, "converted_data")
    state_dim = 25
    control_dim = 8

    batch_process_npz_files(input_folder, output_folder, n=state_dim, m=control_dim)
