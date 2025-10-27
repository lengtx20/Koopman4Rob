"""
This script converts raw CyberDog data into the (x_t, u_t) format, 
and then further into (x_t, u_t, x_{t+1}), saving the results as .npy files.

Batch processing: all .npy files in the current directory will be converted.

Args:
    input_path (str): Path to the input .npy file.
    output_path (str): Path to save the converted .npy file.
    n (int): Dimension of the state.
    m (int): Dimension of the control.
"""


import numpy as np
import os
from data_processor import CyberDogDataProcessor


def convert_xu_to_xux1(data: np.ndarray, n: int, m: int):
    """(x_t, u_t) -> (x_t, u_t, x_t1)"""
    assert data.shape[1] == n + m, f"expected dim {n + m}, but got {data.shape[1]}"
    x_u_t = data[:-1]
    x_tp1 = data[1:, :n]
    xu_x1 = np.concatenate([x_u_t, x_tp1], axis=1)  # shape: (T-1, n + m + n)
    return xu_x1


def batch_process_npy_files(input_dir: str, output_dir: str, n=48, m=6):
    """batch process"""
    os.makedirs(output_dir, exist_ok=True)
    npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    if not npy_files:
        print(f"No .npy file found in {input_dir} ")
        return

    for filename in npy_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            data_processor = CyberDogDataProcessor(input_path)
            data = data_processor.process_data()
            converted_data = convert_xu_to_xux1(data, n, m)
            np.save(output_path, converted_data)
            print(f"[INFO] Converted: {filename}, shape: {converted_data.shape}")
        except Exception as e:
            print(f"[INFO] Failed to convert: {filename}, error: {e}")


if __name__ == "__main__":
    input_folder = os.getcwd()
    output_folder = os.path.join(input_folder, "converted_data")
    state_dim = 48
    control_dim = 6
    batch_process_npy_files(input_folder, output_folder, n=state_dim, m=control_dim)
