import pickle
import numpy as np
import os
import sys
import torch


def load_file(file_path):
    """
    Robustly loads a pkl or npy file.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception:
        try:
            return np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            sys.exit(1)


def compare_data(d1, d2, path="root"):
    """
    Recursively compares two data structures.
    Returns (True, None) if identical, else (False, reason).
    """
    # 1. Check Types
    if type(d1) is not type(d2):
        return False, f"Type mismatch at {path}: {type(d1)} vs {type(d2)}"

    # 2. Numpy Arrays
    if isinstance(d1, np.ndarray):
        if d1.shape != d2.shape:
            return False, f"Shape mismatch at {path}: {d1.shape} vs {d2.shape}"
        if d1.dtype != d2.dtype:
            return False, f"Dtype mismatch at {path}: {d1.dtype} vs {d2.dtype}"

        # Check for NaN handling
        if np.isnan(d1).any() or np.isnan(d2).any():
            if not np.array_equal(np.isnan(d1), np.isnan(d2)):
                return False, f"NaN location mismatch at {path}"
            # Compare non-nan values
            mask = ~np.isnan(d1)
            if not np.array_equal(d1[mask], d2[mask]):
                return False, f"Value mismatch in Numpy Array at {path}"
        else:
            if not np.array_equal(d1, d2):
                return False, f"Value mismatch in Numpy Array at {path}"

    # 3. PyTorch Tensors
    elif torch.is_tensor(d1):
        if d1.shape != d2.shape:
            return False, f"Tensor Shape mismatch at {path}: {d1.shape} vs {d2.shape}"
        if not torch.equal(d1.cpu(), d2.cpu()):
            return False, f"Tensor Value mismatch at {path}"

    # 4. Lists / Tuples
    elif isinstance(d1, (list, tuple)):
        if len(d1) != len(d2):
            return False, f"Length mismatch at {path}: {len(d1)} vs {len(d2)}"
        for i, (item1, item2) in enumerate(zip(d1, d2)):
            is_same, reason = compare_data(item1, item2, path=f"{path}[{i}]")
            if not is_same:
                return False, reason

    # 5. Dictionaries
    elif isinstance(d1, dict):
        if d1.keys() != d2.keys():
            return False, f"Key mismatch at {path}: {d1.keys()} vs {d2.keys()}"
        for key in d1:
            is_same, reason = compare_data(d1[key], d2[key], path=f"{path}['{key}']")
            if not is_same:
                return False, reason

    # 6. Basic Types (int, float, str)
    else:
        if d1 != d2:
            return False, f"Value mismatch at {path}: {d1} != {d2}"

    return True, None


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_pkl.py <file1.pkl> <file2.pkl>")
        print("Example: python compare_pkl.py data_v1.pkl data_v2.pkl")
        return

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    print("=" * 60)
    print(f"Comparing files:\n A: {file1}\n B: {file2}")
    print("=" * 60)

    data1 = load_file(file1)
    data2 = load_file(file2)

    is_identical, reason = compare_data(data1, data2)

    if is_identical:
        print("\n[RESULT] ✅ The files are IDENTICAL.")
    else:
        print("\n[RESULT] ❌ The files are DIFFERENT.")
        print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
