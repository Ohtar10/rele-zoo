from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from relezoo.algorithms.base import Policy
from relezoo.networks.base import Network


class CrossEntropyDiscretePolicy(Policy):
    """Cross Entropy Method - Discrete Policy.

    This policy works with the cross entropy method
    algorithm. The learning step is just a supervised
    learning approach in which the given trajectories
    are expected to be imitated, so a simple
    cross entropy loss is enough to make the policy
    learn.

    """

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
        self.nets = {
            "net": "net.cpt"
        }

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
        """Learn.

        Expects a batch of observations, actions and ignorable weights.
        The network will train via cross entropy loss of predicted
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
        batch_actions = torch.squeeze(batch_actions).to(torch.long).to(self.device)
        self.optimizer.zero_grad()
        pred_actions = self.net(batch_obs)
        loss = self.objective(pred_actions, batch_actions)
        loss.backward()
        self.optimizer.step()
        return loss


