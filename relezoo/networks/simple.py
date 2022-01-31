import torch.nn as nn


class SimpleFC(nn.Module):
    def __init__(self, intput_n: int = 64, output_n: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(intput_n, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_n)
        )

    def forward(self, x):
        return self.net(x)

