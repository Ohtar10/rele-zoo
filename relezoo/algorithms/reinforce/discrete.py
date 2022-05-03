import os.path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from relezoo.algorithms.base import Policy
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode


class ReinforceDiscretePolicy(Policy):
    """Policy
    This class represents a vanilla policy for REINFORCE.
    It is meant to take actions given an observation
    and the underlying neural network.
    Also, it will perform a learning step when provided
    with the necessary objects to calculate the policy
    gradient and perform the backward pass.

    This policy relies on categorical distribution to
    select actions. Hence, this policy only works for
    discrete action spaces.
    """

    def __init__(self,
                 network: Network,
                 learning_rate: float = 1e-2,
                 eps_start: float = 0.0,
                 eps_min: float = 0.0,
                 eps_decay: float = 0.99):
        super(ReinforceDiscretePolicy, self).__init__()
        self.net = network
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.device = "cpu"

    def set_mode(self, mode: NetworkMode):
        if mode == NetworkMode.TRAIN:
            self.net.train()
        else:
            self.net.eval()

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        """act.
        Takes an action given the observation.
        The action will be sampled from a categorical
        distribution considering the logits outputs from
        the underlying neural network."""
        obs = obs.to(self.device)
        if 0.0 < self.eps < np.random.random():
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            out_features = self.net.get_output_shape()
            return np.random.randint(0, out_features)
        else:
            logits = self._get_policy(obs)
            action = logits.sample()
            return action

    def _get_policy(self, obs: torch.Tensor):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        """learn.
        Performs a learning step over the underlying neural
        network using the provided batch of observations, actions, and weights (episode returns)."""
        batch_obs = batch_obs.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_weights = batch_weights.to(self.device)

        self.optimizer.zero_grad()
        batch_loss = self._compute_loss(batch_obs, batch_actions, batch_weights)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss

    def _compute_loss(self, obs, actions, weights):
        """compute loss.
        The loss aka the policy gradient, is just
        the multiplication of the log probabilities
        of all state-action pairs with the weights
        (returns) of the particular episode.

        This loss is equivalent to the gradient
        formula:
        .. math::
            \hat{g} = \frac{1}{|D|}\sum_{\tau \in D}\sum_{t=0}^{T}\nabla_\theta log \pi_\theta (a_t | s_t)R(\tau)

        See https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
        Section: Derivation for Basic Policy Gradient
        """
        logp = self._get_policy(obs).log_prob(actions)
        return -(logp * weights).mean()

    def save(self, save_path: str):
        path = os.path.join(save_path, f"{self.__class__.__name__}.cpt")
        torch.save(self.net, path)

    def load(self, load_path):
        device = "cuda" if self.context and self.context.gpu and torch.cuda.is_available() else "cpu"
        self.net = torch.load(load_path, map_location=torch.device(device))

    def to(self, device: str) -> None:
        self.device = device
        self.net = self.net.to(device)


