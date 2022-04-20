import os
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from relezoo.algorithms.base import Policy, Algorithm
from relezoo.environments.base import Environment
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode
from relezoo.utils.noise import make_noise
from relezoo.utils.structure import Context


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
        self.net = network
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.out_shape = network.get_output_shape()
        log_std = -0.5 * np.ones(self.out_shape, dtype=np.float32)
        # By subscribing the log_std as a parameter, it will receive
        # Gradient updates during the backward pass
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.noise = None
        if noise is not None:
            self._build_noise(noise)
        self.device = "cpu"

    def _build_noise(self, noise: Dict[str, Any]):
        name = noise['name']
        params = noise['params']
        if 'size' in params.keys():
            params['size'] = self.out_shape
        self.noise = make_noise(name, params)

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
        obs = obs.to(self.device)
        distribution = self._get_policy(obs)
        action = distribution.sample()

        if self.noise is not None:
            return action.cpu() + self.noise.sample()
        return action

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: torch.Tensor):
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

    def save(self, save_path: str):
        path = os.path.join(save_path, f"{self.__class__.__name__}.cpt")
        torch.save(self.net, path)

    def to(self, device: str):
        self.device = device
        self.net = self.net.to(device)
        self.log_std = self.log_std.to(device)


class ReinforceContinuous(Algorithm):
    """Reinforce
    Container class for all the necessary logic
    to train and use vanilla policy gradient aka REINFORCE
    with gym environments."""
    def __init__(self,
                 policy: Optional[ReinforceContinuousPolicy] = None,
                 batch_size: int = 5000,
                 logger: Optional[SummaryWriter] = None,
                 play_env: Optional[Environment] = None):
        self.batch_size = batch_size
        self.policy = policy
        self.logger = logger
        self.train_steps = 0
        self.play_env = play_env

    def train(
            self,
            env: Environment,
            context: Context,
            eval_env: Optional[Environment] = None
    ) -> None:
        """train
        The main training loop for the algorithm.

        This is inspired by OpenAI Spinning up implementation.
        """
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.TRAIN)
        episodes = context.episodes
        render = context.render
        device = "cuda" if context.gpu and torch.cuda.is_available() else "cpu"
        self.policy.to(device)
        with tqdm(total=episodes) as progress:
            for i in range(1, episodes + 1):
                is_last_episode = i == episodes
                batch_loss, batch_returns, batch_lens = self._train_epoch(env, self.batch_size)
                if eval_env is not None and (i % context.eval_every == 0 or is_last_episode):  # evaluate every 10 epochs
                    self._evaluate(eval_env, i, render)
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
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lens.append(episode_length)

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

    def _evaluate(self, env: Environment, step: int, render: bool = False):
        render_frames = []
        episode_return = 0
        obs = env.reset()
        while True:
            if render:
                render_frames.append(env.render(mode='rgb_array'))

            action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break

        self.logger.add_scalar("evaluation/return", episode_return, global_step=step)
        self.logger.add_scalar('evaluation/episode_length', len(render_frames), global_step=step)

        if render:
            # T x H x W x C
            sequence = np.array(render_frames)
            # T x C x H x W
            sequence = np.transpose(sequence, [0, 3, 1, 2])
            # B x T x C x H x W
            sequence = np.expand_dims(sequence, axis=0)
            self.logger.add_video("live-play", vid_tensor=sequence, global_step=step, fps=8)

    def _log(self, batch_loss, batch_returns, batch_lens):
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

    def play(self, env: Environment, context: Context) -> (float, int):
        """play.
        Play the environments using the current
        policy, without learning, for as many
        episodes as requested.
        """
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.EVAL)
        episodes = context.episodes
        render = context.render
        device = "cuda" if context.gpu and torch.cuda.is_available() else "cpu"
        self.policy.to(device)
        with tqdm(total=episodes) as progress:
            ep_rewards = []
            ep_lengths = []
            for i in range(1, episodes + 1):
                obs = env.reset()
                ep_length = 1
                ep_reward = 0
                while True:
                    if render:
                        env.render()

                    action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
                    obs, reward, done, _ = env.step(action)
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
