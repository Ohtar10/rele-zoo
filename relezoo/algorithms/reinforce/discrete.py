import os.path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from kink import inject
from tqdm import tqdm

from relezoo.algorithms.base import Policy, Algorithm
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode
from relezoo.utils.structure import Context


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

    def __init__(self, network: Network,
                 learning_rate: float = 1e-2,
                 eps_start: float = 0.0,
                 eps_min: float = 0.0,
                 eps_decay: float = 0.99):
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

    def to(self, device: str) -> None:
        self.device = device
        self.net = self.net.to(device)


@inject
class ReinforceDiscrete(Algorithm):
    """Reinforce
    Container class for all the necessary logic
    to train and use vanilla policy gradient aka REINFORCE
    with gym environments."""

    def __init__(
            self,
            logger: Logging,
            context: Context,
            policy: Optional[ReinforceDiscretePolicy] = None,
            batch_size: int = 5000):
        super(ReinforceDiscrete, self).__init__(context, logger, batch_size, policy)
        self.train_steps = 0

    def _train_epoch(self,
                     env: Environment,
                     batch_size: int = 5000) -> (float, float, int):
        batch_obs = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lens = []

        obs = env.reset()
        episode_rewards = []

        while True:
            batch_obs.append(obs.copy())

            action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, reward, done, _ = env.step(action)

            batch_actions.append(action)
            episode_rewards.append(reward)

            if np.any(done):
                # On episode end, we must build the batch to submit
                # to later make the policy learn.
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lens.append(episode_length)

                # Despite this being called "weights" this is
                # actually R(tau), i.e., the return of the whole
                # trajectory, which is replicated through all the
                # episode steps as this is how the gradient
                # is calculated. See the gradient formula.
                batch_weights += [episode_return] * episode_length

                obs, done, episode_rewards = env.reset(), False, []

                if len(batch_obs) > batch_size:
                    break

        batch_loss = self.policy.learn(
            torch.from_numpy(np.array(batch_obs)),
            torch.from_numpy(np.array(batch_actions)),
            torch.from_numpy(np.array(batch_weights))
        )

        self._log(batch_loss, batch_returns, batch_lens)

        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def _log(self, batch_loss, batch_returns, batch_lens):
        if self.logger is not None:
            self.logger.log_scalar('training/loss', batch_loss, self.train_steps)
            self.logger.log_scalar('training/return', np.mean(batch_returns), self.train_steps)
            self.logger.log_scalar('training/episode_length', np.mean(batch_lens), self.train_steps)
            self.logger.log_grads(self.policy.net, self.train_steps)

    def save(self, save_path: str):
        self.policy.save(save_path)

    def load(self, load_path: str, context: Optional[Context] = None):
        device = "cuda" if context and context.gpu and torch.cuda.is_available() else "cpu"
        net = torch.load(load_path, map_location=torch.device(device))
        self.policy = ReinforceDiscretePolicy(net)
