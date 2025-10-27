import numpy as np


def inspect_npy_file(file_path):
    print(f"\n=== 文件信息: {file_path} ===")
    data = np.load(file_path)

    if isinstance(data, np.lib.npyio.NpzFile):
        print("类型: npz 文件，包含以下键：", list(data.keys()))
        # 自动选择第一个键
        key = list(data.keys())[0]
        arr = data[key]
        print(f"读取键: '{key}' -> 数组形状: {arr.shape}, dtype: {arr.dtype}")
    else:
        print("类型: npy 文件")
        print(f"形状: {data.shape}")
        print(f"dtype: {data.dtype}")
        print("前几行数据:")
        print(data[:3])


if __name__ == "__main__":
    inspect_npy_file("111.npz")
