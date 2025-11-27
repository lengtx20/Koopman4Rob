from mcap_data_loader.utils.basic import get_full_class_name
from typing import Dict
import torch


class LossCalculator:
    def __init__(self, name: str = "MSELoss"):
        self._func = getattr(torch.nn, name)()
        self._name = name

    def __call__(
        self, predictions: torch.Tensor, batch_data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # batch = batch_data["next_state"].squeeze(1)
        # print(f"Norm error: {(predictions - batch).norm() / torch.pi * 180:.4f}")
        # print(f"RMSE error: {((predictions - batch) ** 2).mean().sqrt() / torch.pi * 180:.4f}")
        return self._func(predictions, batch_data["next_state"].squeeze(1))

    def dump(self):
        return {
            "_target_": get_full_class_name(self),
            "name": self._name,
        }
