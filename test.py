import torch

# 创建一个示例矩阵 (3行4列)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])

# 沿着行方向（即对每行）求均值：dim=1
row_means = x.mean(dim=0)

print(row_means)
