"""This file provides the implementation of Deep Koopman model class"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401


def get_activation(name: str):
    return {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "swish": nn.SiLU(),
        "elu": nn.ELU(),
        "mish": nn.Mish(),
        "linear": nn.Identity(),
    }[name]


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
        for in_dim, out_dim in zip(self.sizes[:-1], self.sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=(activation != "linear")))
            if out_dim != self.output_size and activation != "linear":
                layers.append(get_activation(activation))
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
            for in_dim, out_dim in zip(self.sizes[:-1], self.sizes[1:]):
                layers.append(nn.Linear(in_dim, out_dim, bias=(activation != "linear")))
                if out_dim != state_dim and activation != "linear":
                    layers.append(get_activation(activation))
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


class Deep_Koopman(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes,
        lifted_dim,
        activation="relu",
        include_iden_state=True,
        iden_decoder=True,
        requires_grad=True,
        seed=None,
    ):
        """
        state_dim: shape of x
        action_dim: shape of a
        hidden_sizes: hidden layer of the Network in Encoder and Decoder
        lifted_dim: num of dimension of lifted Koopman space
        activation: for the network
        include_iden_state: let the lifted vector to have iden state (first part of the vector)
        """
        super().__init__()
        if seed is not None:
            self.set_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lifted_dim = lifted_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.include_iden_state = include_iden_state
        self.iden_decoder = iden_decoder
        self.requires_grad = requires_grad

        self.encoder = Encoder(
            state_dim,
            action_dim,
            hidden_sizes,
            lifted_dim,
            activation,
            include_iden_state,
        )
        self.decoder = Decoder(
            state_dim,
            action_dim,
            hidden_sizes,
            lifted_dim,
            activation,
            include_iden_state,
            iden_decoder,
        )
        self.init_matrix()

        for param in self.parameters():
            print(type(param), param.size())
            param.requires_grad = requires_grad

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def init_matrix(self):
        self.A = nn.Parameter(torch.empty(self.lifted_dim, self.lifted_dim))
        self.B = nn.Parameter(torch.empty(self.lifted_dim, self.action_dim))
        # He initialization
        nn.init.kaiming_uniform_(self.A, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.B, a=0, mode="fan_in", nonlinearity="relu")

    def to_device(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.A.data = self.A.data.to(device)
        self.B.data = self.B.data.to(device)

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
        return self.A @ z.T + self.B @ u.T

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, get_action: bool = False
    ) -> torch.Tensor:
        """Predict next state: x -> z -> z_next -> x_next"""
        z = self.encode(x)
        # print(z.shape, u.shape)
        z_next = self.linear_dynamics(z, u)
        return self.decode(z_next, get_action)

    def freeze_matrix(self):
        pass

    def freeze_encoder(self):
        pass

    def freeze_decoder(self):
        pass

    def save(self, model_dir):
        torch.save(self.encoder.state_dict(), f"{model_dir}/encoder.pth")
        torch.save(self.decoder.state_dict(), f"{model_dir}/decoder.pth")
        torch.save(self.A.data, f"{model_dir}/A.pth")
        torch.save(self.B.data, f"{model_dir}/B.pth")

    def load(self, model_dir):
        self.encoder.load_state_dict(torch.load(f"{model_dir}/encoder.pth"))
        self.decoder.load_state_dict(torch.load(f"{model_dir}/decoder.pth"))
        self.A.data = torch.load(f"{model_dir}/A.pth")
        self.B.data = torch.load(f"{model_dir}/B.pth")

    def __repr__(self):
        return (
            f"Deep_Koopman(state_dim={self.state_dim}, action_dim={self.action_dim}, lifted_dim={self.lifted_dim}, "
            f"hidden_sizes={self.hidden_sizes}, activation={self.activation}, "
            f"include_iden_state={self.include_iden_state}, iden_decoder={self.iden_decoder})"
        )
