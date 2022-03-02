from typing import Optional

import os
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from relezoo.algorithms.base import Policy, Algorithm
from relezoo.environments.base import Environment
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode


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

    def __init__(self, network: Network, learning_rate: float = 1e-2):
        self.net = network
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.out_shape = network.get_output_shape()
        log_std = -0.5 * np.ones(self.out_shape, dtype=np.float32)
        # By subscribing the log_std as a parameter, it will receive
        # Gradient updates during the backward pass
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def set_mode(self, mode: NetworkMode):
        if mode == NetworkMode.TRAIN:
            self.net.train()
        else:
            self.net.eval()

    def _get_policy(self, obs: torch.Tensor):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        """act.
        Takes an action given the observation.
        The action will be sampled from a normal
        distribution considering mu and std output
        from the underlying neural network."""
        distribution = self._get_policy(obs)
        action = distribution.sample().item()
        return action

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: torch.Tensor):
        """learn.
        Performs a learning step over the underlying neural
        network using the provided batch of observations, actions, and weights (episode returns)."""
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

    def save(self, save_path: str):
        path = os.path.join(save_path, f"{self.__class__.__name__}.cpt")
        torch.save(self.net, path)


class ReinforceContinuous(Algorithm):
    """Reinforce
    Container class for all the necessary logic
    to train and use vanilla policy gradient aka REINFORCE
    with gym environments."""
    def __init__(self, env: Environment, policy: Optional[ReinforceContinuousPolicy] = None,
                 logger: Optional[SummaryWriter] = None):
        self.env = env
        self.obs_space = env.get_observation_space()[0]
        self.act_space = env.get_action_space()[0]
        self.policy = policy
        self.logger = logger
        self.train_steps = 0

    def train(self, episodes: int = 50, batch_size: int = 5000, render: bool = False) -> None:
        """train
        The main training loop for the algorithm.

        This is inspired by OpenAI Spinning up implementation.
        """
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.TRAIN)
        with tqdm(total=episodes) as progress:
            for i in range(1, episodes + 1):
                is_last_episode = i == episodes
                batch_loss, batch_returns, batch_lens = self._train_epoch(batch_size, render, is_last_episode)
                progress.set_postfix({
                    "loss": f"{batch_loss:.2f}",
                    "score": f"{np.mean(batch_returns):.2f}",
                    "episode_length": f"{np.mean(batch_lens):.2f}"
                })
                progress.update()
                if self.logger is not None:
                    self.logger.flush()
        if self.logger is not None:
            self.logger.close()

    def _train_epoch(self,
                     batch_size: int = 5000,
                     render: bool = False,
                     is_last_episode: bool = False) -> (float, float, int):
        batch_obs = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lens = []

        obs = self.env.reset()
        episode_rewards = []
        render_episode = False
        render_frames = []

        while True:
            if render and not render_episode:
                self.env.render()

            if not render_episode:
                render_frames.append(self.env.render(mode='rgb_array'))

            batch_obs.append(obs.copy())

            action = self.policy.act(torch.from_numpy(obs))
            obs, reward, done, _ = self.env.step([action])
            batch_actions.append(action)
            episode_rewards.append(reward)

            if done:
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lens.append(episode_length)

                batch_weights += [episode_return] * episode_length

                obs, done, episode_rewards = self.env.reset(), False, []

                render_episode = True

                if len(batch_obs) > batch_size:
                    break

        batch_loss = self.policy.learn(
            torch.from_numpy(np.array(batch_obs)),
            torch.from_numpy(np.array(batch_actions)),
            torch.from_numpy(np.array(batch_weights))
        )

        self._log(is_last_episode, batch_loss, batch_returns, batch_lens, render_frames)
        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def _log(self, is_last_episode: bool, batch_loss, batch_returns, batch_lens, render_frames):
        if self.logger is not None:
            self.logger.add_scalar('loss', batch_loss, self.train_steps)
            self.logger.add_scalar('return', np.mean(batch_returns), self.train_steps)
            self.logger.add_scalar('episode_length', np.mean(batch_lens), self.train_steps)

        for name, p in self.policy.net.named_parameters():
            if p.grad is None:
                continue
            name = name.replace(".", "/")
            self.logger.add_histogram(
                tag=f"grads/{name}", values=p.grad.detach().cpu().numpy(), global_step=self.train_steps
            )

        if self.policy.log_std.grad is not None:
            p = self.policy.log_std
            self.logger.add_histogram(
                tag="grads/log_std", values=p.grad.detach().cpu().numpy(), global_step=self.train_steps
            )

        if render_frames and (self.train_steps % 10 == 0 or is_last_episode):
            sequence = np.array(render_frames)
            sequence = np.transpose(sequence, [0, 3, 1, 2])
            sequence = np.expand_dims(sequence, axis=0)
            tag = 'end-training' if is_last_episode else 'training'
            self.logger.add_video(tag, vid_tensor=sequence, global_step=self.train_steps, fps=8)

    def play(self, episodes: int, render: bool = False) -> (float, int):
        """play.
        Play the environment using the current
        policy, without learning, for as many
        episodes as requested.
        """
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.EVAL)
        with tqdm(total=episodes) as progress:
            ep_rewards = []
            ep_lengths = []
            for i in range(1, episodes + 1):
                obs = self.env.reset()
                ep_length = 1
                ep_reward = 0
                while True:
                    if render:
                        self.env.render()

                    action = self.policy.act(torch.from_numpy(obs))
                    obs, reward, done, _ = self.env.step([action])
                    ep_reward += reward
                    if done:
                        break
                    ep_length += 1
                ep_lengths.append(ep_length)
                ep_rewards.append(ep_reward)
                progress.set_postfix({
                    "Avg. reward": f"{np.mean(ep_rewards):.2f}",
                    "Avg. ep length": f"{np.mean(ep_length):.2f}"
                })
                progress.update()
        return np.mean(ep_rewards), np.mean(ep_length)

    def save(self, save_path: str) -> None:
        self.policy.save(save_path)

    def load(self, load_path: str) -> None:
        net = torch.load(load_path)
        self.policy = ReinforceContinuousPolicy(net)
