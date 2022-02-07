import torch.nn as nn


class SimpleFC(nn.Module):
    """SimpleFC.

    This is a simple & static fully
    connected neural network of two
    hidden layers. It is meant for
    very simple tasks and is not
    designed to handle complex or dynamic
    algorithms.
    """
    def __init__(self, in_shape: int = 64, out_shape: int = 2):
        super().__init__()
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

