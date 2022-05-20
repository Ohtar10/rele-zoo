from typing import Optional, Dict, Any

import torch
import torch.optim as optim

from relezoo.algorithms.base import Policy
from relezoo.networks.base import Network
from relezoo.utils.noise import make_noise


class ReinforceContinuousPolicy(Policy):
    """Policy
    This class represents a vanilla policy for REINFORCE.
    It is meant to take actions given an observation
    and the underlying neural network.
    Also, it will perform a learning step when provided
    with the necessary objects to calculate the policy
    gradient and perform the backward pass.

    This policy is designed to work for
    continuous action spaces, i.e., it needs to learn
    the median and standard deviation of a normal
    distribution from which the continuous action
    will be sampled.
    """

    def __init__(self,
                 network: Network,
                 learning_rate: float = 1e-2,
                 noise: Optional[Dict[str, Any]] = None):
        super(ReinforceContinuousPolicy, self).__init__()
        self.net = network
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.out_shape = network.get_output_shape()
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

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        """act.
        Takes an action given the observation.
        The action will be sampled from a normal
        distribution considering mu and std output
        from the underlying neural network."""
        obs = obs.to(self.device)
        distribution = self._get_policy(obs)
        action = distribution.sample()

        if self.noise is not None:
            return action.cpu() + self.noise.sample()
        return action

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

    def _compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, rtau: torch.Tensor):
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
        return -(logp * rtau).mean()

