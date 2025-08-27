import numpy as np

data_name = "s_v_s1/"
data_list = []
for i in range(64):
    res_data = np.load(f"resnet_process/{i}.npy")
    state_data = np.load(f"state/{i}.npy")
    x = state_data[:-1, :]
    x1 = state_data[1:, :]
    a = res_data[:-1, :]
    data = np.concatenate((x, a, x1), axis=1)
    print(f"[INFO] Data shape for {i}: {data.shape}")
    np.save(f"s_v_s1/{i}.npy", data)
