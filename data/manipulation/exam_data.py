import numpy as np

def main():
    num_files = 64
    resnet_means, resnet_vars = [], []
    state_means, state_vars = [], []

    # 用于全局拼接
    all_resnet = []
    all_state = []

    for i in range(num_files):
        res_data = np.load(f"resnet_process/{i}.npy")
        state_data = np.load(f"state/{i}.npy")

        # 保存到全局列表
        all_resnet.append(res_data.reshape(-1))
        all_state.append(state_data.reshape(-1))

        # 每个文件单独统计
        resnet_means.append(np.mean(res_data))
        resnet_vars.append(np.var(res_data))
        state_means.append(np.mean(state_data))
        state_vars.append(np.var(state_data))

        print(f"[INFO] File {i}: "
              f"ResNet mean={resnet_means[-1]:.6f}, var={resnet_vars[-1]:.6f} | "
              f"State mean={state_means[-1]:.6f}, var={state_vars[-1]:.6f}")

    # ===== 文件级别统计结果 =====
    print("\n[SUMMARY - File Level]")
    print(f"ResNet: mean of means={np.mean(resnet_means):.6f}, "
          f"mean of vars={np.mean(resnet_vars):.6f}")
    print(f"State : mean of means={np.mean(state_means):.6f}, "
          f"mean of vars={np.mean(state_vars):.6f}")

    # ===== 全局统计（拼接后整体计算） =====
    all_resnet = np.concatenate(all_resnet, axis=0)
    all_state = np.concatenate(all_state, axis=0)

    print("\n[SUMMARY - Global Level]")
    print(f"ResNet: global mean={np.mean(all_resnet):.6f}, global var={np.var(all_resnet):.6f}")
    print(f"State : global mean={np.mean(all_state):.6f}, global var={np.var(all_state):.6f}")

if __name__ == "__main__":
    main()
