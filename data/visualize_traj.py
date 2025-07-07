import torch
import matplotlib.pyplot as plt

# 系统参数
A = torch.tensor([[0.9, 0.1], [-0.2, 0.95]])
B = torch.tensor([[0.1], [0.05]])
T = 100  # 时间步长
state_dim = 2
action_dim = 1
x = torch.randn(state_dim)  # 初始状态

xs = []
us = []

for t in range(T):
    u = torch.randn(action_dim) * 0.5  # 控制输入
    x_next = torch.tanh(x @ A.T + u @ B.T)
    xs.append(x.numpy())
    us.append(u.numpy())
    x = x_next

xs = torch.tensor(xs)
us = torch.tensor(us)

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(xs[:, 0], label='x1')
plt.plot(xs[:, 1], label='x2')
plt.title("State Trajectory")
plt.xlabel("Time step")
plt.ylabel("State")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(us[:, 0], label='u')
plt.title("Control Input")
plt.xlabel("Time step")
plt.ylabel("u")
plt.legend()

plt.tight_layout()
plt.show()
