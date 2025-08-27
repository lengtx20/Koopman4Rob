import numpy as np

data_name = "s_v_s1"
data_list = []
for i in range(49):
    v_and_s = np.load(f"{i}.npy")
    vision_data = v_and_s[:, :10]
    state_data = v_and_s[:, 10:]
    s = state_data[:-1, :]
    s1 = state_data[1:, :]
    v = vision_data[:-1, :]

    data = np.concatenate((s, v, s1), axis=1)
    print(f"[INFO] Data shape for {i}: {data.shape}")
    np.save(f"{i}.npy", data)
