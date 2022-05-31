from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch import optim

from relezoo.algorithms.base import Policy
from relezoo.networks.base import Network
from relezoo.utils.noise import make_noise


class CrossEntropyContinuousPolicy(Policy):
    """Cross Entropy Method - Continuous Policy.

    This policy works with the cross entropy method
    algorithm. The learning step is just a supervised
    learning approach in which the given trajectories
    are expected to be imitated.

    The network is mean to predict mu and sigma of
    a normal distribution from which an action
    will be sampled and returned. Because of this,
    the loss function in this case is the MSEloss
    for continuous values.

    """

    def __init__(self,
                 network: Network,
                 learning_rate: float = 1e-2,
                 noise: Optional[Dict[str, Any]] = None):
        super(CrossEntropyContinuousPolicy, self).__init__()
        self.net = network
        self.objective = nn.MSELoss()
        self.out_shape = network.get_output_shape()
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.noise = None
        if noise is not None:
            self._build_noise(noise)
        self.nets = {
            "net": "net.cpt"
        }

    def _build_noise(self, noise: Dict[str, Any]):
        name = noise['name']
        params = noise['params']
        if 'size' in params.keys():
            params['size'] = self.out_shape
        self.noise = make_noise(name, params)

    def _get_policy(self, obs: torch.Tensor):
        mu, sigma = self.net(obs)
        return torch.distributions.Normal(mu, sigma)

    def act(self, obs: torch.Tensor, explore: bool = False) -> (torch.Tensor, int):
        obs = obs.to(self.device)
        distribution = self._get_policy(obs)
        action = distribution.sample()

        if explore and self.noise is not None:
            return action.cpu() + self.noise.sample()
        return action

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        """Learn.

        Expects a batch of observations, actions and ignorable weights.
        The network will train via MSE loss of predicted continuous
        actions against the observations and the provided actions as
        ground truth.

        Parameters
        ----------
        batch_obs : torch.Tensor
            Batch of observations
        batch_actions : torch.Tensor
            Batch of actions taken per observation
        batch_weights : Optional[torch.Tensor]
            Ignored.

        """
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
