import numpy as np


def inspect_npz_shapes_for_xhand(log_path):
    data = np.load(log_path)
    print(f"Loaded npz file: {log_path}\n")

    for key in data.files:
        value = data[key]
        print(f"Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")


def extract_flattened_data_for_xhand(file_path):
    data = np.load(file_path)

    positions = data["positions"][:, :8]  # shape: (85, 8)
    torques = data["torques"][:, :8]  # shape: (85, 8)
    calc_force = data["calc_force"][:, :3, :].reshape(
        85, -1
    )  # shape: (85, 3, 3) -> (85, 9)
    # temperature = data["temperature"][:, :3, :].reshape(85, -1) # shape: (85, 3, 20) -> (85, 60)
    # calc_temperature = data["calc_temprature"][:, :3]           # shape: (85, 5)
    finger_pos_command = data["finger_pos_command"][:, :8]  # shape:(85, 8)

    all_data = np.concatenate(
        [
            positions,
            torques,
            calc_force,
            finger_pos_command,
        ],
        axis=1,
    )  # shape: (85, 8+8+9+8=33)

    return all_data
