import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from koopman4rob.models.deep_koopman import DeepKoopman


class KoopmanDataset(Dataset):
    """Simple wrapper to load numpy data into PyTorch DataLoader."""

    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KoopmanMathAnalyzer:
    """
    A formalized analyzer for the Koopman Operator.
    This class computes the mathematical quantities (G, P, K) defined in Controlled EDMD.
    """

    def __init__(self, model: DeepKoopman, device="cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir):
        """Loads weights from the specified directory."""
        self.model.load(model_dir)
        print(f"[INFO] Model loaded from {model_dir}")

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """
        The lifting function \psi(x).
        Wraps the model's encoder to map state x -> feature z.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            z = self.model.encode(x)  # identity map is handled inside
        return z

    def compute_GP_matrices(self, data_loader: DataLoader):
        """
        Computes the Extended Gram Matrix G and Interaction Matrix P for Controlled Koopman.

        Definitions:
            Theta_t = [z_t, a_t]  (Extended State)
            G = (1/M) * Sum (Theta_t * Theta_t^T)
            P = (1/M) * Sum (Theta_t * z_{t+1}^T)
            K_opt^T = G_dagger * P  (where K_opt = [A, B])

        Returns:
            G (np.array): Extended Gram Matrix (L+m x L+m)
            P (np.array): Interaction Matrix (L+m x L)
            K_opt (np.array): Optimal Operator (L+m x L) -> Represents [A^T; B^T]
        """
        self.model.eval()

        lifted_dim = self.model.lifted_dim
        action_dim = self.model.action_dim
        total_dim = lifted_dim + action_dim

        # Accumulators
        G_acc = torch.zeros(total_dim, total_dim, device=self.device)
        P_acc = torch.zeros(total_dim, lifted_dim, device=self.device)
        M_total = 0

        print("[INFO] Accumulating G and P matrices (Controlled Formulation)...")
        print(
            f"       Extended Dim: {total_dim} (Lifted: {lifted_dim} + Action: {action_dim})"
        )

        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = batch.to(self.device)
                state_dim = self.model.state_dim

                # Parse Batch: [x_t, a_t, x_{t+1}]
                x_t = batch[:, :state_dim]
                u_t = batch[:, state_dim : state_dim + action_dim]
                x_t1 = batch[:, -state_dim:]

                # 1. Lift States
                z_t = self.psi(x_t)
                z_t1 = self.psi(x_t1)

                # 2. Construct Extended State Theta = [z, u]
                Theta_t = torch.cat([z_t, u_t], dim=1)

                # 3. Accumulate
                # G = Theta^T * Theta
                G_acc += Theta_t.T @ Theta_t
                # P = Theta^T * z_{t+1}
                P_acc += Theta_t.T @ z_t1

                M_total += batch.size(0)

        # Normalize
        G = G_acc / M_total
        P = P_acc / M_total

        # Solve for K_opt (The "Perfect" Matrix for these features)
        # Add epsilon for numerical stability
        reg = 1e-8 * torch.eye(total_dim, device=self.device)
        K_opt = torch.linalg.pinv(G + reg) @ P

        return G.cpu().numpy(), P.cpu().numpy(), K_opt.cpu().numpy()

    def analyze_eigenvalues(self, K_matrix=None):
        """
        Computes and plots eigenvalues of the operator.
        If K_matrix is provided, it must be the square state transition matrix A.
        """
        if K_matrix is None:
            vals = torch.linalg.eigvals(self.model.A.data).cpu().numpy()
            label = "Trained Model A"
        else:
            vals = np.linalg.eigvals(K_matrix)
            label = "Analytical A"

        # Sort by magnitude
        idx = np.argsort(np.abs(vals))[::-1]
        vals = vals[idx]

        plt.figure(figsize=(6, 6))
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Unit Circle")
        plt.scatter(vals.real, vals.imag, c="red", marker="x", label="Eigenvalues")
        plt.title(f"Spectral Analysis: {label}")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.show()

        return vals
