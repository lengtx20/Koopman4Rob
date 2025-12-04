"""This file provides the implementation of Deep Koopman model class"""

import torch
import torch.nn as nn
from class_resolver.contrib.torch import activation_resolver
from typing import Dict, Optional, List, Literal
from pathlib import Path
from pydantic import NonNegativeInt, PositiveInt
from basis import DictBatch
from mcap_data_loader.basis.cfgable import InitConfigMixin
from mcap_data_loader.utils.array_like import get_device_auto
from mcap_data_loader.utils.basic import DataBasicConfig


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
        identity=True,
    ):
        """iden_decoder: if set True, the decoder will be a simple slice of the lifted vector. C = I."""
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lifted_dim = lifted_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.identity = identity

        if not self.identity:
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

    def forward(self, z):
        if self.identity:
            decoded = z[:, : self.state_dim]
        else:
            decoded = self.layers(z)
        return decoded

    def __repr__(self):
        return (
            f"Decoder(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"lifted_dim={self.lifted_dim}, hidden_sizes={self.hidden_sizes}, "
            f"activation={self.activation}, identity={self.identity})"
        )


class DeepKoopmanConfig(DataBasicConfig):
    """Configuration for the model."""

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

    def model_post_init(self, context):
        if self.iden_decoder and not self.include_iden_state:
            raise ValueError(
                "If `iden_decoder` is True, `include_iden_state` must also be True."
            )


class DeepKoopman(nn.Module, InitConfigMixin):
    def __init__(self, config: DeepKoopmanConfig):
        super().__init__()
        self.config = config
        self._config_name = "config.yaml"

    def add_first_batch(self, batch: DictBatch) -> None:
        if not batch:
            return
        # get the state and action dims
        config = self.config
        state_dim = (
            config.state_dim if config.state_dim > 0 else batch["cur_state"].shape[-1]
        )
        action_dim = (
            config.action_dim
            if config.action_dim > 0
            else batch["cur_action"].shape[-1]
        )
        self.config = config.model_copy(
            update={"state_dim": state_dim, "action_dim": action_dim}
        )
        self.get_logger().info(f"State dim: {state_dim}, Action dim: {action_dim}")

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

    def encode(self, x) -> torch.Tensor:
        """x -> z"""
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor:
        """z -> x"""
        return self.decoder(z)

    def linear_dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """z -> z_next = A * z + B * u"""
        # return (self.A @ z.T + self.B @ u.T).T
        # for debugging
        z_item = self.A @ z.T
        u_item = self.B @ u.T
        state_dim = self.config.state_dim
        z_state = z_item.squeeze(1)[:state_dim]
        u_delta = u_item.squeeze(1)[:state_dim]
        z_delta = z_state - z.squeeze(0)[:state_dim]
        z_delta_norm = torch.norm(z_delta)
        u_delta_norm = torch.norm(u_delta)
        self.get_logger().info(
            f"{z_delta=}, {u_delta=}\n"
            f"{z_delta_norm.item()=:.6f}, {u_delta_norm.item()=:.6f}"
        )
        return (z_item + u_item).T

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict next state: x -> z -> z_next -> x_next"""
        z = self.encode(batch["cur_state"].squeeze(1))
        z_next = self.linear_dynamics(z, batch["cur_action"].squeeze(1))
        # the prediction dim should be (B, T, D)
        # unsqueeze to make T=1
        return self.decode(z_next).unsqueeze(1)

    def save(self, path: Path):
        torch.save(self.encoder.state_dict(), path / "encoder.pth")
        torch.save(self.A.data, path / "A.pth")
        torch.save(self.B.data, path / "B.pth")
        if not self.config.iden_decoder:
            torch.save(self.decoder.state_dict(), path / "decoder.pth")
        self.save_config(path / self._config_name)

    def load(self, path: Optional[Path] = None):
        # TODO: configure the ns
        with torch.device(get_device_auto("torch", self.config.device)):
            return self._load(path)

    def _load(self, path: Optional[Path] = None):
        """Load weights while preserving the current device and dtype."""
        # TODO: make use of the dtype config
        if path is not None:
            self = type(self)(path / self._config_name)
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
        return self

    def __repr__(self):
        return self.config.__repr__()
