#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_csv_stats.py
---------------------------------------
用于快速检查 TensorBoard 导出 CSV 文件的数值情况。

功能：
- 显示行数、列数
- 显示 "Value" 列的最小值、最大值
- 计算后半段 Value 的均值与标准差

用法：
    python check_csv_stats.py your_file.csv
"""

import sys
import pandas as pd
import numpy as np

def analyze_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return

    if "Value" not in df.columns:
        print("⚠️ 文件中未找到列名 'Value'，请确认是 TensorBoard 导出的 CSV。")
        return

    values = df["Value"].to_numpy()
    n = len(values)

    print(f"\n✅ 文件: {csv_path}")
    print(f"总行数: {n}")
    print(f"Value 范围: min = {values.min():.4f}, max = {values.max():.4f}")

    # 后半段统计
    half = n // 2
    tail_values = values[half:]
    mean_tail = np.mean(tail_values)
    std_tail = np.std(tail_values)

    print(f"\n📊 后半段统计 ({n - half} 个样本):")
    print(f"平均值 (mean): {mean_tail:.4f}")
    print(f"标准差 (std): {std_tail:.4f}")

    # 可选：显示最后几行（检查趋势）
    print("\n最后 5 行数据示例:")
    print(df.tail())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_csv_stats.py your_file.csv")
    else:
        analyze_csv(sys.argv[1])
