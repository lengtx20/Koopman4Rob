import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class DynamicSystemDataset(Dataset):
    '''generate a simple uncontrolled dynamic system dataset'''
    def __init__(self, num_samples=1000, noise_std=0.1):
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.data = self.generate_demo_data()

    def generate_demo_data(self):
        data = torch.randn(self.num_samples, 1) 
        next_state = data**2 + torch.randn_like(data) * self.noise_std
        self.data = torch.cat((data, next_state), dim=1)    # [x(t), x(t+1)]

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]  # x(t), x(t+1)

class ControlledDynamicSystemDataset(Dataset):
    '''generate a controlled dynamic system dataset'''
    def __init__(self, num_samples=1000, noise_std=0.1, state_dim=2, action_dim=1, T=100):
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T
        self.data = self.generate_demo_data()

    def generate_demo_data(self):
        A = torch.tensor([[0.9, 0.1], [-0.2, 0.95]])  # 状态转移矩阵
        B = torch.tensor([[0.1], [0.05]])             # 控制输入矩阵
        data = torch.randn(self.num_samples, self.state_dim)  # 初始状态
        trajectory_data = []

        for i in range(self.num_samples):
            x = data[i]
            traj = [x]
            for t in range(self.T):
                u = torch.randn(self.action_dim) * 0.5  # 控制输入
                x_next = torch.tanh(x @ A.T + u @ B.T)  # 动力学更新
                traj.append(x_next)
                x = x_next
            trajectory_data.append(torch.stack(traj))  # 每个样本的轨迹

        self.data = torch.stack(trajectory_data)  # [num_samples, T, state_dim]
        return self.data

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]  # 返回轨迹

    def visualize_trajectory(self, idx):
        trajectory = self.data[idx].numpy()  # 获取一条轨迹
        plt.figure(figsize=(12, 6))

        # 绘制状态轨迹
        plt.subplot(1, 2, 1)
        plt.plot(trajectory[:, 0], label='x1')
        plt.plot(trajectory[:, 1], label='x2')
        plt.title(f"State Trajectory (Sample {idx})")
        plt.xlabel("Time step")
        plt.ylabel("State")
        plt.legend()

        # 绘制控制输入轨迹（假设控制输入是随时间变化的）
        plt.subplot(1, 2, 2)
        control_inputs = torch.randn(self.T, self.action_dim) * 0.5  # 随机控制输入
        plt.plot(control_inputs[:, 0], label='u')
        plt.title(f"Control Input Trajectory (Sample {idx})")
        plt.xlabel("Time step")
        plt.ylabel("Control Input")
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dataset = ControlledDynamicSystemDataset(num_samples=10, noise_std=0.1)
    dataset.visualize_trajectory(0)




