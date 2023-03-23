# from warnings import warn
# import numpy as np
# import scipy
# import tensorflow as tf
# from sklearn.utils.validation import check_is_fitted
# todo: write a lightweight Koopman package in pytorch
from __future__ import annotations

import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

activations_dict = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "swish": nn.SiLU(),
    "elu": nn.ELU(),
    "mish": nn.Mish(),
    "linear": nn.Identity(),
}


class FFNN(nn.Module):
    """A feedforward neural network with customizable architecture and
    activation functions.

    Args:
        input_size (int): The size of the input layer.
        hidden_sizes (list): A list of the sizes of the hidden layers.
        output_size (int): The size of the output layer.
        activations (str): A string for activation functions for every
        layer.

    Attributes:
        layers (nn.ModuleList): A list of the neural network layers.
    """

    def __init__(self, input_size, hidden_sizes, output_size, activations):
        super(FFNN, self).__init__()

        # Define the activation
        act = activations_dict[activations]

        # Define the input layer
        self.layers = nn.ModuleList()
        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, output_size))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(act)

            # Define the hidden layers
            for i in range(1, len(hidden_sizes)):
                self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                self.layers.append(act)

            # Define the output layer
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        """Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The output tensor of the neural network.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class StableKMatrix(torch.nn.Module):
    """
    A PyTorch module for creating trainable skew-symmetric matrices
    with negative diagonal elements.

    Args:
        dim (int): The dimension of the matrix.
        init_std (float): The standard deviation for initializing
        the trainable parameters
        # with truncated normal distribution.

    Attributes:
        diagonal (torch.nn.Parameter): The diagonal elements of the
        skew-symmetric matrix.
        off_diagonal (torch.nn.Parameter): The off-diagonal elements
        of the skew-symmetric matrix.
    """

    def __init__(self, dim: int, init_std: float = 0.1):
        super().__init__()
        self.dim = dim

        # diagonal part = - some ^ 2
        self.diagonal = torch.nn.Parameter(
            -torch.pow(torch.nn.init.trunc_normal_(torch.empty(dim), std=init_std), 2)
        )
        self.off_diagonal = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(dim, dim), std=init_std)
        )

    def forward(self):
        # Construct skew-symmetric matrix
        skew_symmetric = torch.diag(self.diagonal)
        skew_symmetric -= self.off_diagonal - self.off_diagonal.t()

        return skew_symmetric


class DLKoopmanRegressor(nn.Module):
    def __init__(self, config_encoder, config_decoder, config_koopman):
        super(DLKoopmanRegressor, self).__init__()

        self._encoder = FFNN(
            input_size=config_encoder["input_size"],
            hidden_sizes=config_encoder["hidden_sizes"],
            output_size=config_encoder["output_size"],
            activations=config_encoder["activations"],
        )

        self._decoder = FFNN(
            input_size=config_decoder["input_size"],
            hidden_sizes=config_decoder["hidden_sizes"],
            output_size=config_decoder["output_size"],
            activations=config_decoder["activations"],
        )

        self._state_matrix = StableKMatrix(
            dim=config_encoder["output_size"],
            init_std=config_koopman["init_std"],
        )

    def forward(self, x):
        encoded = self._encoder(x)
        state_matrix = self._state_matrix(encoded)
        decoded = self._decoder(state_matrix)
        return decoded


class NNDMD(object):
    """Implementation of Neural Network DMD"""

    def __init__(self, config_encoder, config_decoder, config_koopman):
        # build self.regressor, following `.dmd`
        self.DLKoopmanRegressor = DLKoopmanRegressor(
            config_encoder, config_decoder, config_koopman
        )
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.config_koopman = config_koopman

    def fit(self, x, y=None, dt=None):
        # compute forward

        # compile the loss function

        # run optimization
        pass

    def predict(self, x):
        pass

    @property
    def coef_(self):
        pass

    @property
    def state_matrix_(self):
        return self._state_matrix_

    @property
    def eigenvalues_(self):
        pass

    @property
    def unnormalized_modes(self):
        pass

    def compute_eigen_phi(self, x):
        pass


# if __name__ == "__main__":
#     config_encoder = {"input_size": 2, "hidden_sizes": [3, 3],
#                       "output_size": 4, "activations": "tanh"}
#     config_decoder = {"input_size": 4, "hidden_sizes": [],
#                       "output_size": 2, "activations": "linear"}
#     config_koopman = {"init_std": 0.1}
#
#     model = NNDMD(config_encoder, config_decoder, config_koopman)
#
#     from torchsummary import summary
#
#     summary(model._encoder, input_size=(2,), batch_size=-1)
#     summary(model._decoder, input_size=(4,), batch_size=-1)
