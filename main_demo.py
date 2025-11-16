"""This file provides a demo main scirpt for training with Deep Koopman and computing EWC weights"""

from runner.koopman_runner import KoopmanRunner
import os
import pickle 
from config import Config
from hydra_zen import instantiate, store
import hydra
import logging
import torch

def load_data(mode: str, data_path: str, ratio: float = 0.8):
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        return None, None

    # ============ load data ===============
    data = pickle.load(open(data_path, 'rb'))
    data_tensor = torch.from_numpy(data).float()  # (N, 108)

    print(f"[INFO] Loaded data → Tensor: {data_tensor.shape}")

    N = data_tensor.shape[0]
    if N < 2:
        return None, None

    state_dim, action_dim = 48, 12

    # ============ create episodes =============
    episode = [
        {
            "cur_state":  data_tensor[t, :state_dim].unsqueeze(0),      # (1,48)
            "cur_action":     data_tensor[t, state_dim:state_dim+action_dim].unsqueeze(0),  # (1,12)
            "next_state": data_tensor[t+1, :state_dim].unsqueeze(0),    # (1,48)
        }
        for t in range(N - 1)
    ]

    # =========== Divide train and val =============
    split = int(len(episode) * ratio)
    train_ep = episode[:split]
    val_ep = episode[split:]

    print(f"[INFO] Train steps: {len(train_ep)}, Val steps: {len(val_ep)}")
    return [train_ep], [val_ep]





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
def run(config: Config):
    stage = config.stage
    train_data,val_data = load_data(stage,'./koopman_data/plane_20251116_175525.pkl' )
    runner = KoopmanRunner(config, train_data,val_data)
    # stage = config.stage
    model_dir = config.checkpoint_path
    logger.info(f"{stage=}, {model_dir=}")
    if stage == "train":
        # automatically change model dir if exists
        if model_dir.exists():
            ids = [int(p.stem) for p in model_dir.parent.iterdir() if p.stem.isdigit()]
            next_id = max(ids) + 1 if ids else 0
            model_dir = model_dir.parent / f"{next_id}{model_dir.suffix}"
        model_dir.mkdir(parents=True)
        config.checkpoint_path = model_dir
    runner.run(stage)


def main():
    config_name = "class_config"
    store(Config, name=config_name)
    store.add_to_hydra_store()

    config_path = "configs"
    config_name = "config"
    config = None

    def convert(cfg):
        nonlocal config
        config = instantiate(cfg)

    hydra.main(config_path or None, config_name, None)(convert)()
    if config:
        run(Config(**config))


if __name__ == "__main__":
    main()

