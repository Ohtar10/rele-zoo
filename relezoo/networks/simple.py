from typing import Any

import torch.nn as nn

from relezoo.networks.base import Network


class SimpleFC(Network):
    """Simple Fully Connected Neural Network.

    This is a simple & static fully
    connected neural network of two
    hidden layers. It is meant for
    very simple tasks and is not
    designed to handle complex or dynamic
    algorithms.

    Network:

    .. code-block::

        nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, out_shape)
        )

    """

    def __init__(self, in_shape: int = 64, out_shape: int = 2):
        """

        Parameters
        ----------
        in_shape : int
            Input shape
        out_shape : int
            Output shape
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, out_shape)
        )

    def forward(self, x):
        return self.net(x)

    def get_input_shape(self) -> Any:
        return self.in_shape

    def get_output_shape(self) -> Any:
        return self.out_shape


class SimpleFCMuSigma(Network):
    """Simple Fully Connected Neural Network for mu and sigma.

    This follows the same architecture as
    :py:class:`relezoo.networks.simple.SimpleFC`
    but its forward pass predicts the mu and sigma
    of a normal distribution.

    This network is meant to be used with
    continuous action spaces environments.

    Network:

    .. code-block::

        self.base = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.mu_head = nn.Linear(64, out_shape)
        self.sigma_head = nn.Sequential(
            nn.Linear(64, out_shape),
            nn.Softplus()
        )

    """

    def __init__(self, in_shape: int, out_shape: int):
        """

        Parameters
        ----------
        in_shape : int
            Input shape
        out_shape : int
            Output shape
        """
        super(SimpleFCMuSigma, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.base = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.mu_head = nn.Linear(64, out_shape)
        self.sigma_head = nn.Sequential(
            nn.Linear(64, out_shape),
            nn.Softplus()
        )

    def forward(self, x: Any) -> Any:
        x = self.base(x)
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return mu, sigma

    def get_input_shape(self) -> Any:
        return self.in_shape

    def get_output_shape(self) -> Any:
        return self.out_shape
