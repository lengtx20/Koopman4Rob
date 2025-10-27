"""
Implements Elastic Weight Consolidation (EWC) for the Deep Koopman model.

Features:
- Fisher information matrix computation and storage
- Mean parameter snapshot saving/loading
- EWC regularization penalty computation
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class EWCModel:
    def __init__(self, model, device):
        """
        Args:
            model (torch.nn.Module): Trained model whose parameters to regularize.
            device (torch.device): Device for computation.
        """
        self.model = model
        self.device = device

        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        self.means = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher = None

    def compute_fisher(self, data_tensor, batch_size=64):
        """
        Compute Fisher Information Matrix approximation based on Koopman reconstruction loss.

        Args:
            data_tensor (torch.Tensor): (N, state+action+next_state)
            batch_size (int): batch size
        Returns:
            dict: Fisher information for each parameter
        """
        self.model.eval()
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        fisher_dict = {
            n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()
        }

        print("[EWC] Computing Fisher Information Matrix...")
        for batch in tqdm(loader, desc="Fisher"):
            x = batch[0].to(self.device)
            state_dim = self.model.state_dim
            act_dim = self.model.action_dim

            x_t = x[:, :state_dim]
            a_t = x[:, state_dim : state_dim + act_dim]
            x_t1 = x[:, -state_dim:]

            self.model.zero_grad()
            pred = self.model(x_t, a_t, training=False)
            loss = 0.5 * torch.mean((pred - x_t1) ** 2)
            loss.backward()

            for n, p in self.params.items():
                if p.grad is not None:
                    fisher_dict[n] += p.grad.detach() ** 2

        for n in fisher_dict.keys():
            fisher_dict[n] /= len(loader)
            fisher_dict[n] = torch.clamp(fisher_dict[n], min=1e-8)

        self.fisher = fisher_dict
        print("[EWC] Fisher computation finished.")
        return fisher_dict

    def penalty(self, model):
        """Compute the EWC penalty for the given model."""
        if self.fisher is None:
            raise ValueError("[EWC] Fisher matrix not computed or loaded.")

        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                fisher = self.fisher[n]
                mean = self.means[n]
                loss += torch.sum(fisher * (p - mean) ** 2)
        return loss

    def save(self, save_dir, task_id=0):
        os.makedirs(save_dir, exist_ok=True)
        fisher_path = os.path.join(save_dir, f"fisher_task{task_id}.pt")
        mean_path = os.path.join(save_dir, f"mean_task{task_id}.pt")

        torch.save(self.fisher, fisher_path)
        torch.save(self.means, mean_path)
        print(f"[EWC] Saved fisher → {fisher_path}")
        print(f"[EWC] Saved means → {mean_path}")

    def load(self, fisher_path, mean_path=None):
        """Load Fisher and means from disk."""
        if not os.path.exists(fisher_path):
            raise FileNotFoundError(f"[EWC] Missing fisher file: {fisher_path}")

        self.fisher = torch.load(fisher_path, map_location=self.device)
        print(f"[EWC] Loaded Fisher from {fisher_path}")

        if mean_path is not None and os.path.exists(mean_path):
            self.means = torch.load(mean_path, map_location=self.device)
            print(f"[EWC] Loaded parameter means from {mean_path}")
        else:
            print("[EWC] No mean file provided; using model snapshot.")
