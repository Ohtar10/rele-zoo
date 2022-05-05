import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from relezoo.algorithms.base import Policy
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode


class CrossEntropyDiscretePolicy(Policy):

    def __init__(self,
                 network: Network,
                 learning_rate: float = 1e-2,
                 eps_start: float = 0.0,
                 eps_min: float = 0.0,
                 eps_decay: float = 0.99):
        super(CrossEntropyDiscretePolicy, self).__init__()
        self.net = network
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.device = 'cpu'

    def set_mode(self, mode: NetworkMode):
        if mode == NetworkMode.TRAIN:
            self.net.train()
        else:
            self.net.eval()

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        if 0.0 < self.eps < np.random.random():
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            out_features = self.net.get_output_shape()
            batch_size = obs.shape[0]
            actions = np.random.randint(0, out_features, batch_size)
            return torch.tensor(actions)

        obs = obs.to(self.device)
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits).sample()

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        batch_obs = batch_obs.to(self.device)
        batch_actions = torch.squeeze(batch_actions).to(torch.long).to(self.device)
        self.optimizer.zero_grad()
        pred_actions = self.net(batch_obs)
        loss = self.objective(pred_actions, batch_actions)
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, save_path):
        path = os.path.join(save_path, f"{self.__class__.__name__}.cpt")
        torch.save(self.net, path)

    def load(self, load_path):
        device = "cuda" if self.context and self.context.gpu and torch.cuda.is_available() else "cpu"
        self.net = torch.load(load_path, map_location=torch.device(device))

    def to(self, device: str) -> None:
        self.device = device
        self.net = self.net.to(device)


