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
    This class computes the mathematical quantities (G, A, K) defined in EDMD
    using the features learned by a trained DeepKoopman model.
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
        # Ensure input is tensor and on device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            z = self.model.encode(x)  # identitcal map is handled inside
        return z

    def compute_formal_matrices(self, data_loader: DataLoader):
        """
        Computes the matrices G and A exactly as defined in the formalization.

        Equations:
            G = (1/M) * \sum \psi(x_t)* \psi(x_t)
            A = (1/M) * \sum \psi(x_t)* \psi(x_{t+1})
            K = G^{\conj-T} A

        Note: This formulation finds the best linear operator K such that z_{t+1} \psi z_t * K
        ignoring the control input u. This represents the autonomous evolution
        projected onto the feature space.

        Returns:
            G (np.array): Gram matrix (Covariance of features)
            A (np.array): Data interaction matrix
            K_est (np.array): Estimated Koopman Operator via Least Squares
        """
        self.model.eval()

        # Accumulators for G and A
        # Dimensions: L x L where L is lifted_dim
        G_acc = torch.zeros(
            self.model.lifted_dim, self.model.lifted_dim, device=self.device
        )
        A_acc = torch.zeros(
            self.model.lifted_dim, self.model.lifted_dim, device=self.device
        )
        M_total = 0

        print("[INFO] Accumulating matrices G and A over dataset...")

        with torch.no_grad():
            for batch in tqdm(data_loader):
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)

                # Extract x_t and x_{t+1}
                # Assuming batch format: [x_t, a_t, x_{t+1}]
                state_dim = self.model.state_dim
                action_dim = self.model.action_dim

                x_t = batch[:, :state_dim]
                # We skip a_t for the purely autonomous matrix calculation defined in the prompt
                x_t1 = batch[:, -state_dim:]

                # 1. Compute Lifted States
                # \psi_m        (Batch x Lifted)
                Psi_t = self.psi(x_t)
                # \psi_{m+1}    (Batch x Lifted)
                Psi_t1 = self.psi(x_t1)

                # 2. Accumulate Outer Products
                # G += \psi_t.T @ \psi_t
                G_acc += Psi_t.T @ Psi_t
                # A += \psi_t.T @ \psi_{t+1}
                A_acc += Psi_t.T @ Psi_t1

                M_total += x_t.size(0)

        # 3. Normalize by M
        G = G_acc / M_total
        A = A_acc / M_total

        # 4. Compute K = G_dagger @ A
        # Using pinverse for stability
        G_inv = torch.linalg.pinv(G)
        K_est = G_inv @ A

        return G.cpu().numpy(), A.cpu().numpy(), K_est.cpu().numpy()

    def compute_controlled_matrices(self, data_loader: DataLoader):
        """
        Computes the EDMD matrices considering Control inputs.
        This corresponds to finding A and B such that z_{t+1} = K * (z_t A^T + u_t B^T).

        Extended State \psi_c = [\psi(x), u]

        Returns:
            G_ext: Gram matrix of extended state
            A_ext: Cross-correlation
            K_full: [A^T, B^T]^T matrix
        """
        self.model.eval()

        lifted_dim = self.model.lifted_dim
        action_dim = self.model.action_dim
        total_dim = lifted_dim + action_dim

        G_acc = torch.zeros(total_dim, total_dim, device=self.device)
        A_acc = torch.zeros(total_dim, lifted_dim, device=self.device)
        M_total = 0

        print("[INFO] Computing Extended EDMD (with Control)...")
        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = batch.to(self.device)
                state_dim = self.model.state_dim

                x_t = batch[:, :state_dim]
                u_t = batch[:, state_dim : state_dim + action_dim]
                x_t1 = batch[:, -state_dim:]

                z_t = self.psi(x_t)
                z_t1 = self.psi(x_t1)

                # Construct Extended Regressor [z_t, u_t]
                # Shape: (Batch, Lifted + Action)
                theta_t = torch.cat([z_t, u_t], dim=1)

                # Accumulate
                G_acc += theta_t.T @ theta_t
                A_acc += theta_t.T @ z_t1

                M_total += batch.size(0)

        G_ext = G_acc / M_total
        A_cross = A_acc / M_total

        # Solve for K_full (shape: Total_In x Lifted_Out)
        # K_full^T = [A_dense, B_dense]
        K_full = torch.linalg.pinv(G_ext) @ A_cross

        return G_ext.cpu().numpy(), A_cross.cpu().numpy(), K_full.cpu().numpy()

    def analyze_eigenvalues(self, K_matrix=None):
        """
        Computes and plots eigenvalues of the operator.
        If K_matrix is provided (numpy), uses that.
        Otherwise uses the trained model's A matrix.
        """
        if K_matrix is not None:
            # Assumes K_matrix follows z_{t+1} = z_t @ K
            # Transpose to get standard operator form z_{t+1} = K_std @ z_t
            # if the K_matrix was computed as weight matrix W (z' = z W).
            # The compute_formal_matrices returns W such that row_vec_next = row_vec @ W.
            # Eigenvalues of W and W.T are identical.
            vals = np.linalg.eigvals(K_matrix)
            label = "Analytical K"
        else:
            # Model A is stored such that z' = z A^T + ...
            # So the operator matrix is A (or A.T). Eigs are same.
            vals = torch.linalg.eigvals(self.model.A.data).cpu().numpy()
            label = "Trained Model A"

        # Sort by magnitude
        idx = np.argsort(np.abs(vals))[::-1]
        vals = vals[idx]

        # Plot
        plt.figure(figsize=(6, 6))
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5, label="Unit Circle")

        plt.scatter(
            vals.real, vals.imag, c="red", marker="x", label=f"Eigenvalues ({label})"
        )
        plt.title(f"Spectral Analysis: {label}")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.show()

        max_rad = np.max(np.abs(vals))
        print(f"[{label}] Max Spectral Radius: {max_rad:.4f}")
        return vals

    def compare_operators(self, data_loader):
        """
        Compares the trained Matrix A vs the analytically computed Matrix K.
        This tells you how close the gradient descent solution is to the optimal linear solution
        given the *current* fixed features.
        """
        print("\n--- Comparing Trained vs Analytical Operators ---")

        # 1. Get Analytical K (Extended)
        _, _, K_full_est = self.compute_extended_matrices(data_loader)

        # K_full_est is shape (Lifted+Action, Lifted)
        # It contains [A_opt^T; B_opt^T]
        L = self.model.lifted_dim
        A_opt_T = K_full_est[:L, :]
        B_opt_T = K_full_est[L:, :]

        # 2. Get Trained Matrices
        # Model stores A, B such that forward is z @ A.T + u @ B.T
        # So model.A is actually A_model (Target x Source) if standard linear layer
        # Let's check deep_koopman.py:
        #   self.A = nn.Parameter(torch.empty(lifted, lifted))
        #   linear_dynamics: (self.A @ z.T + ...).T  => z @ A.T
        # So self.A is indeed the operator matrix (or close to it).

        A_model = self.model.A.data.cpu().numpy()  # Shape (L, L)
        B_model = self.model.B.data.cpu().numpy()  # Shape (L, Action)

        # Note: The computed K_full_est is for Row Vector multiplication: z_{t+1} = [z, u] @ K
        # So K_full_est columns are the destination features.
        # K_full_est row i maps source feature i to all destinations.
        # Therefore, A_opt_T matches shape (L, L) and corresponds to A_model.T

        diff_A = np.linalg.norm(A_model.T - A_opt_T) / np.linalg.norm(A_opt_T)
        diff_B = np.linalg.norm(B_model.T - B_opt_T) / np.linalg.norm(B_opt_T)

        print(f"Rel. Error (A_trained vs A_analytical): {diff_A:.6f}")
        print(f"Rel. Error (B_trained vs B_analytical): {diff_B:.6f}")

        return A_opt_T, B_opt_T
