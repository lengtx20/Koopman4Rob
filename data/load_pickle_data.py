import pickle
import numpy as np
from models.process_image import EncodeImage


def load_pickle_data(data_path: str) -> np.ndarray:
    with open(data_path, "rb") as f:
        data = pickle.load(f)

        print(len(data))
        ep_data: dict = data[0]

        print(ep_data.keys())

        print(ep_data["observations"].keys())
        print(ep_data["observations"]["wrist"].shape)
        print(ep_data["observations"]["env_close"].shape)
        print(ep_data["actions"].shape)

        ep_data_used: list = data[:60]

        # The structure of the data need to be (num_sample, x_t + a_t + x_t1)
        # 7 state dims, 512 action dims, 7 next state dims
        state_dim = 7
        action_dim = 512 * 2
        samples_used = np.zeros(
            (len(ep_data_used) - 1, state_dim + action_dim + state_dim),
            dtype=np.float32,
        )
        image_encoder = EncodeImage()
        for i in range(len(ep_data_used) - 1):
            ep = ep_data_used[i]
            obs = ep["observations"]
            cam_names = ["wrist", "env_close"]
            state = ep["actions"]
            next_state = ep_data_used[i + 1]["actions"]
            samples_used[i, :state_dim] = state
            for index, cam_name in enumerate(cam_names):
                cam_arr = obs[cam_name]
                cam_feat = image_encoder.encode_image(cam_arr)
                samples_used[
                    i, state_dim + index * 512 : state_dim + (index + 1) * 512
                ] = cam_feat
            samples_used[i, -state_dim:] = next_state

        print(samples_used.shape)

        # assert samples_train[-1, :state_dim] == state
        # assert samples_train[-1, -state_dim:] == next_state
        print(samples_used[-1, :state_dim])
        print(state)
        assert np.allclose(samples_used[-1, :state_dim], state)
        print(samples_used[-1, -state_dim:])
        print(next_state)
        assert np.allclose(samples_used[-1, -state_dim:], next_state)

        assert np.allclose(
            samples_used[-1, state_dim + 512 : state_dim + 512 + 512], cam_feat
        )

        return samples_used
