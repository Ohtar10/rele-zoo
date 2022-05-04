from typing import Optional

import torch
import torch.nn as nn
import os
from torch import optim

from relezoo.algorithms.base import Policy
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode


class CrossEntropyContinuousPolicy(Policy):

    def __init__(self,
                 network: Network,
                 learning_rate: float = 1e-2):
        super(CrossEntropyContinuousPolicy, self).__init__()
        self.net = network
        self.objective = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.device = 'cpu'

    def _get_policy(self, obs: torch.Tensor):
        mu, sigma = self.net(obs)
        return torch.distributions.Normal(mu, sigma)

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        obs = obs.to(self.device)
        distribution = self._get_policy(obs)
        action = distribution.sample()
        return action

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        batch_obs = batch_obs.to(self.device)
        batch_actions = torch.squeeze(batch_actions).to(self.device)
        self.optimizer.zero_grad()
        distribution = self._get_policy(batch_obs)
        pred_log_prob = distribution.log_prob(distribution.sample())
        old_log_prob = distribution.log_prob(batch_actions.unsqueeze(dim=-1))
        loss = self.objective(pred_log_prob, old_log_prob)
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, save_path):
        path = os.path.join(save_path, f"{self.__class__.__name__}.cpt")
        torch.save(self.net, path)

    def load(self, load_path):
        device = "cuda" if self.context and self.context.gpu and torch.cuda.is_available() else "cpu"
        self.net = torch.load(load_path, map_location=torch.device(device))

    def set_mode(self, mode: NetworkMode):
        if mode == NetworkMode.TRAIN:
            self.net.train()
        else:
            self.net.eval()

    def to(self, device: str):
        self.device = device
        self.net = self.net.to(device)
