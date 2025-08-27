import numpy as np
import argparse
import os


def inspect_npy_file(file_path, show_examples=True, num_examples=3):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"\n=== 文件信息: {file_path} ===")
    print(f"类型: {type(data)}")
    print(f"数据类型 dtype: {data.dtype}")
    if isinstance(data, np.ndarray):
        print(f"数据形状 shape: {data.shape}")
        print(f"维度 ndim: {data.ndim}")
    else:
        print(f"不是 ndarray，类型为 {type(data)}，无法显示 shape。")

    if data.dtype == "object":
        print("\n 注意：这是对象数组，数据中可能包含字典、列表或其他复杂结构。")
        if isinstance(data[0], dict):
            print(f"字段示例：{list(data[0].keys())}")
        else:
            print(f"元素类型：{type(data[0])}")

    if show_examples:
        print(f"\n=== 前 {num_examples} 个数据项预览 ===")
        for i in range(min(len(data), num_examples)):
            print(f"\n--- 第 {i} 个元素 ---")
            print(f"类型: {type(data[i])}")
            print(data[i])

    print("\n=== 分析完成 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 .npy 文件结构")
    parser.add_argument("file", type=str, help="要分析的 .npy 文件路径")
    parser.add_argument("--no-preview", action="store_true", help="是否关闭样本预览")
    parser.add_argument("--num", type=int, default=3, help="要显示的样本个数")
    args = parser.parse_args()

    inspect_npy_file(
        args.file, show_examples=not args.no_preview, num_examples=args.num
    )
