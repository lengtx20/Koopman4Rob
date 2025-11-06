"""This file provides the implementation of Deep Koopman model class"""

import torch
import torch.nn as nn
import json
from class_resolver.contrib.torch import activation_resolver
from typing import Dict, Optional, List, Literal
from pathlib import Path
from pydantic import BaseModel, NonNegativeInt, PositiveInt, ConfigDict
from data.mcap_data_utils import DictBatch


class Encoder(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes,
        lifted_dim,
        activation,
        include_iden_state=True,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lifted_dim = lifted_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.include_iden_state = include_iden_state
        # lifted dim is the final output dimension of the encoder
        self.output_size = lifted_dim - state_dim if include_iden_state else lifted_dim

        layers = []
        self.sizes = [self.state_dim] + self.hidden_sizes + [self.output_size]
        # last linear layer index
        last_i = len(self.sizes) - 2
        not_linear = activation != "linear"
        for i, (in_dim, out_dim) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim, bias=(not_linear or i == last_i)))
            if not_linear and i != last_i:
                layers.append(activation_resolver.make(activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.layers(x)  # state_dim -> output_size
        output = torch.cat([x, encoded], dim=1) if self.include_iden_state else encoded
        return output  # lifted_dim

    def __repr__(self):  # print the architecture of the encoder
        return (
            f"Encoder(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"lifted_dim={self.lifted_dim}, hidden_sizes={self.hidden_sizes}, "
            f"activation={self.activation}, include_iden_state={self.include_iden_state})"
        )


class Decoder(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes,
        lifted_dim,
        activation,
        include_iden_state=True,
        iden_decoder=True,
    ):
        """iden_decoder: if set True, the decoder will be a simple slice of the lifted vector. C = I."""
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lifted_dim = lifted_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.include_iden_state = include_iden_state
        self.iden_decoder = iden_decoder

        if not self.iden_decoder:
            layers = []
            self.sizes = [self.lifted_dim] + hidden_sizes + [state_dim]
            last_i = len(self.sizes) - 2
            not_linear = activation != "linear"
            for i, (in_dim, out_dim) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
                layers.append(
                    nn.Linear(in_dim, out_dim, bias=(not_linear or i == last_i))
                )
                if not_linear and i != last_i:
                    layers.append(activation_resolver.make(activation))
            self.layers = nn.Sequential(*layers)

    def forward(self, z, get_action=False):
        """
        If get_action is true, the return of decoder will also contain a self-defined a_t1.
        This is to enhance the flexibility when doing multi-step prediction.
        This part needs to be modified by user.
        """
        if self.include_iden_state and self.iden_decoder:
            decoded = z[:, : self.state_dim]
        else:
            decoded = self.layers(z)
        return (decoded, 0) if get_action else decoded  # action: i.e. RL policy

    def __repr__(self):  # print the architecture of the decoder
        return (
            f"Decoder(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"lifted_dim={self.lifted_dim}, hidden_sizes={self.hidden_sizes}, "
            f"activation={self.activation}, include_iden_state={self.include_iden_state}, "
            f"iden_decoder={self.iden_decoder})"
        )


class DeepKoopmanConfig(BaseModel):
    """Configuration for the model."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_dim: NonNegativeInt = 0
    """Dimension of the system state. If 0, will be inferred from data."""
    action_dim: NonNegativeInt = 0
    """Dimension of the control input. If 0, will be inferred from data."""
    hidden_sizes: List[PositiveInt]
    """Sizes of hidden layers in the model."""
    lifted_dim: PositiveInt
    """Dimension of the lifted space."""
    activation: str = "relu"
    """Activation function to use in the model."""
    include_iden_state: bool = True
    """Whether to include identity state in the model."""
    iden_decoder: bool = True
    """Whether to use identity decoder in the model."""
    threshold_mode: Optional[Literal["neural_ratio", "ewc_loss"]] = None
    """Mode for EWC thresholding."""
    fisher_path: Optional[Path] = None
    """Path to save/load Fisher information matrix for EWC."""


class DeepKoopman(nn.Module):
    def __init__(self, config: DeepKoopmanConfig):
        super().__init__()
        self.config = config

    def add_first_batch(self, batch: DictBatch) -> None:
        # get the state and action dims
        state_dim = batch["cur_state"].shape[1]
        action_dim = batch["cur_action"].shape[1]
        print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")
        config = self.config
        if config.state_dim == 0:
            config.state_dim = state_dim
        if config.action_dim == 0:
            config.action_dim = action_dim

    def _init_matrix(self, a=None, b=None):
        lifted_dim = self.config.lifted_dim
        action_dim = self.config.action_dim
        # only initialize when no pre-trained matrix is provided
        a_data = torch.empty(lifted_dim, lifted_dim) if a is None else a
        b_data = torch.empty(lifted_dim, action_dim) if b is None else b
        self.A = nn.Parameter(a_data)
        self.B = nn.Parameter(b_data)
        if a is None:
            nn.init.kaiming_uniform_(self.A, a=0, mode="fan_in", nonlinearity="relu")
        if b is None:
            nn.init.kaiming_uniform_(self.B, a=0, mode="fan_in", nonlinearity="relu")

    def clip_A_spectral_radius(self, max_radius=1.0):
        """Clips the spectral radius of matrix A to a maximum value"""
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(
                self.A, full_matrices=False
            )  # SVD: A = U * S * V^T
            S_clipped = torch.clamp(S, max=max_radius)
            self.A.data = (U @ torch.diag(S_clipped) @ Vh).to(self.A.device)

    def encode(self, x) -> torch.Tensor:
        """x -> z"""
        return self.encoder(x)

    def decode(self, z, get_action=False):
        """z -> x"""
        return self.decoder(z, get_action)

    def linear_dynamics(self, z: torch.Tensor, u: torch.Tensor):
        """z -> z_next = A * z + B * u"""
        return (self.A @ z.T + self.B @ u.T).T

    def forward(
        self, batch: Dict[str, torch.Tensor], get_action: bool = False
    ) -> torch.Tensor:
        """Predict next state: x -> z -> z_next -> x_next"""
        z = self.encode(batch["cur_state"])
        z_next = self.linear_dynamics(z, batch["cur_action"])
        return self.decode(z_next, get_action)

    def freeze_matrix(self):
        pass

    def freeze_encoder(self):
        pass

    def freeze_decoder(self):
        pass

    def save(self, path: Path):
        torch.save(self.encoder.state_dict(), path / "encoder.pth")
        torch.save(self.A.data, path / "A.pth")
        torch.save(self.B.data, path / "B.pth")
        if not self.config.iden_decoder:
            torch.save(self.decoder.state_dict(), path / "decoder.pth")
        json.dump(self.config.model_dump(), open(path / "hparams.json", "w"))

    def load(self, path: Optional[Path] = None):
        """Load weights while preserving the current device and dtype."""
        if path is not None:
            try:
                self.config = DeepKoopmanConfig(
                    **json.load(open(path / "hparams.json", "r"))
                )
            except Exception as e:
                print(e)
        config = self.config
        self.encoder = Encoder(
            config.state_dim,
            config.action_dim,
            config.hidden_sizes,
            config.lifted_dim,
            config.activation,
            config.include_iden_state,
        )
        self.decoder = Decoder(
            config.state_dim,
            config.action_dim,
            config.hidden_sizes,
            config.lifted_dim,
            config.activation,
            config.include_iden_state,
            config.iden_decoder,
        )
        if path is not None:
            encoder_state = torch.load(path / "encoder.pth")
            self.encoder.load_state_dict(encoder_state)
            a = torch.load(path / "A.pth")
            b = torch.load(path / "B.pth")
            # TODO: should always iden?
            if not config.iden_decoder:
                decoder_state = torch.load(path / "decoder.pth")
                self.decoder.load_state_dict(decoder_state)
        else:
            a, b = None, None
        self._init_matrix(a, b)
        if config.threshold_mode is not None:
            self.load_fisher(fisher_path=config.fisher_path)
        self.device = torch.get_default_device()
        return self

    def __repr__(self):
        return self.config.__repr__()

    def load_fisher(self, fisher_path, task_id=1):
        self.ckpt = torch.load(fisher_path)
        self.fisher_dict = self.ckpt.get("fisher_dict", {})
        print("[INFO] fisher_dict length:", len(self.fisher_dict))
        print("[INFO] fisher_dict keys:", self.fisher_dict.keys())
        if isinstance(self.fisher_dict, dict) and isinstance(
            list(self.fisher_dict.values())[0], dict
        ):
            self.fisher_dict = list(self.fisher_dict.values())[task_id - 1]

    def register_gradient_masks(self, threshold_mode, ewc_threshold):
        def create_mask(fisher_tensor):
            with torch.device(self.device):
                mask = torch.ones_like(fisher_tensor)
                if threshold_mode == "value":
                    mask[fisher_tensor < ewc_threshold] = 0
                elif threshold_mode == "neural_ratio":
                    thresh_val = torch.quantile(fisher_tensor.view(-1), ewc_threshold)
                    mask[fisher_tensor < thresh_val] = 0
                elif threshold_mode == "weight_ratio":
                    min_val = fisher_tensor.min()
                    max_val = fisher_tensor.max()
                    thresh_val = min_val + ewc_threshold * (max_val - min_val)
                    mask[fisher_tensor < thresh_val] = 0
                else:
                    raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
                return mask

        # -------- param A --------
        param_A = self.A
        fisher_A = self.fisher_dict.get("A", None)
        if fisher_A is not None and fisher_A.shape == param_A.shape:
            mask_A = create_mask(fisher_A)
            param_A.register_hook(lambda grad: grad * mask_A)
        else:
            print("[Warning] No valid Fisher info for A")

        # -------- param B --------
        param_B = self.B
        fisher_B = self.fisher_dict.get("B", None)
        if fisher_B is not None and fisher_B.shape == param_B.shape:
            mask_B = create_mask(fisher_B)
            param_B.register_hook(lambda grad: grad * mask_B)
        else:
            print("[Warning] No valid Fisher info for B")

        # -------- encoder.layers.0.weight --------
        param_0_w = self.encoder.layers[0].weight
        fisher_0_w = self.fisher_dict.get("encoder.layers.0.weight", None)
        if fisher_0_w is not None and fisher_0_w.shape == param_0_w.shape:
            mask_0_w = create_mask(fisher_0_w)
            param_0_w.register_hook(lambda grad: grad * mask_0_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.0.weight")

        # -------- encoder.layers.0.bias --------
        param_0_b = self.encoder.layers[0].bias
        fisher_0_b = self.fisher_dict.get("encoder.layers.0.bias", None)
        if fisher_0_b is not None and fisher_0_b.shape == param_0_b.shape:
            mask_0_b = create_mask(fisher_0_b)
            param_0_b.register_hook(lambda grad: grad * mask_0_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.0.bias")

        # -------- encoder.layers.2.weight --------
        param_2_w = self.encoder.layers[2].weight
        fisher_2_w = self.fisher_dict.get("encoder.layers.2.weight", None)
        if fisher_2_w is not None and fisher_2_w.shape == param_2_w.shape:
            mask_2_w = create_mask(fisher_2_w)
            param_2_w.register_hook(lambda grad: grad * mask_2_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.2.weight")

        # -------- encoder.layers.2.bias --------
        param_2_b = self.encoder.layers[2].bias
        fisher_2_b = self.fisher_dict.get("encoder.layers.2.bias", None)
        if fisher_2_b is not None and fisher_2_b.shape == param_2_b.shape:
            mask_2_b = create_mask(fisher_2_b)
            param_2_b.register_hook(lambda grad: grad * mask_2_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.2.bias")

        # -------- encoder.layers.4.weight --------
        param_4_w = self.encoder.layers[4].weight
        fisher_4_w = self.fisher_dict.get("encoder.layers.4.weight", None)
        if fisher_4_w is not None and fisher_4_w.shape == param_4_w.shape:
            mask_4_w = create_mask(fisher_4_w)
            param_4_w.register_hook(lambda grad: grad * mask_4_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.4.weight")

        # -------- encoder.layers.4.bias --------
        param_4_b = self.encoder.layers[4].bias
        fisher_4_b = self.fisher_dict.get("encoder.layers.4.bias", None)
        if fisher_4_b is not None and fisher_4_b.shape == param_4_b.shape:
            mask_4_b = create_mask(fisher_4_b)
            param_4_b.register_hook(lambda grad: grad * mask_4_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.4.bias")

        # -------- encoder.layers.6.weight --------
        param_6_w = self.encoder.layers[6].weight
        fisher_6_w = self.fisher_dict.get("encoder.layers.6.weight", None)
        if fisher_6_w is not None and fisher_6_w.shape == param_6_w.shape:
            mask_6_w = create_mask(fisher_6_w)
            param_6_w.register_hook(lambda grad: grad * mask_6_w)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.6.weight")

        # -------- encoder.layers.6.bias --------
        param_6_b = self.encoder.layers[6].bias
        fisher_6_b = self.fisher_dict.get("encoder.layers.6.bias", None)
        if fisher_6_b is not None and fisher_6_b.shape == param_6_b.shape:
            mask_6_b = create_mask(fisher_6_b)
            param_6_b.register_hook(lambda grad: grad * mask_6_b)
        else:
            print("[Warning] No valid Fisher info for encoder.layers.6.bias")
