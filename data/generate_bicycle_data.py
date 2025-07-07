import numpy as np

def generate_data_concat(num_samples=10000, dt=0.1):
    state_dim = 4
    action_dim = 2
    
    x_range = [-10, 10]
    y_range = [-10, 10]
    theta_range = [-np.pi, np.pi]
    v_range = [0, 10]
    acc_range = [-3, 3]
    steering_range = [-0.5, 0.5]
    
    data = []
    for _ in range(num_samples):
        x_t = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            np.random.uniform(*theta_range),
            np.random.uniform(*v_range),
        ])
        a_t = np.array([
            np.random.uniform(*acc_range),
            np.random.uniform(*steering_range),
        ])
        x_tp1 = bicycle_model_step(x_t, a_t, dt)
        
        sample = np.concatenate([x_t, a_t, x_tp1], axis=0)
        data.append(sample)
    
    return np.array(data)  # shape: (num_samples, 10)

def bicycle_model_step(x, a, dt=0.1, L=2.0):
    px, py, theta, v = x
    acc, steering_rate = a
    
    px_next = px + v * np.cos(theta) * dt
    py_next = py + v * np.sin(theta) * dt
    theta_next = theta + (v / L) * np.tan(steering_rate) * dt
    v_next = v + acc * dt
    
    return np.array([px_next, py_next, theta_next, v_next])

if __name__ == "__main__":
    dataset = generate_data_concat(10000, dt=0.1)
    print("数据形状:", dataset.shape)  # (10000, 10)
    # 保存成npy文件
    np.save("koopman_bicycle_dataset.npy", dataset)
    print("数据已保存为 koopman_bicycle_dataset.npy")
