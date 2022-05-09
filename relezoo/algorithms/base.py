from abc import ABC, abstractmethod
from typing import Optional, Any

import numpy as np
import torch
from kink import inject
from tqdm import tqdm

from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.network import NetworkMode
from relezoo.utils.structure import Context


@inject
class Policy(ABC):
    """Policy.

    Represents a policy of an on-policy RL algorithm.
    """

    def __init__(self, context: Context = None):
        self.context = context

    @abstractmethod
    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        pass

    @abstractmethod
    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @abstractmethod
    def load(self, load_path):
        pass

    @abstractmethod
    def set_mode(self, mode: NetworkMode):
        pass

    @abstractmethod
    def to(self, device: str):
        pass


class Algorithm(ABC):
    """Algorithm.

    Represents an RL algorithm to solve a task.
    """

    def __init__(self,
                 context: Context,
                 logging: Logging,
                 batch_size: int,
                 policy: Optional[Policy] = None):
        self.context = context
        self.logger = logging
        self.policy = policy
        self.batch_size = batch_size
        self.mean_reward_window = 100
        self.avg_return_pool = []
        self.train_steps = 0

    def train(self, env: Environment, eval_env: Optional[Environment] = None) -> Any:
        """train
        The main training loop for the algorithm.

        This is inspired by OpenAI Spinning up implementation.
        """
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.TRAIN)
        start_at_step = self.context.start_at_step
        epochs = start_at_step + self.context.epochs
        render = self.context.render
        device = "cuda" if self.context.gpu and torch.cuda.is_available() else "cpu"
        self.policy.to(device)
        with tqdm(initial=start_at_step, total=epochs) as progress:
            for i in range(start_at_step + 1, epochs + 1):
                is_last_epoch = i == epochs
                batch_loss, batch_returns, batch_lens = self.train_epoch(env, self.batch_size)
                if eval_env is not None and (i % self.context.eval_every == 0 or is_last_epoch):
                    self.evaluate(eval_env, True if render and i % self.context.render_every == 0 else False)
                progress.set_postfix({
                    "loss": f"{batch_loss:.2f}",
                    "mean_batch_score": f"{np.mean(batch_returns):.2f}",
                    "mean_batch_ep_length": f"{np.mean(batch_lens):.2f}",
                    f"mean_reward_{self.mean_reward_window}": f"{np.mean(self.avg_return_pool):.2f}"
                })
                progress.update()
                if self.logger is not None:
                    self.logger.flush()
        if self.logger is not None:
            self.logger.close()

        mean_reward = np.mean(self.avg_return_pool)
        return f"Training finished -- Mean reward over {self.mean_reward_window} epochs: " \
               f"{mean_reward:.2f}", mean_reward

    @abstractmethod
    def train_epoch(self, env: Environment, batch_size: int):
        pass

    def evaluate(self, env: Environment, render: bool = False):
        render_frames = []
        episode_return = 0
        episode_length = 0
        obs = env.reset()
        while True:
            if render:
                render_frames.append(env.render(mode='rgb_array'))

            action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            if done:
                break

        self.avg_return_pool.append(episode_return)
        self.logger.log_scalar("evaluation/return", episode_return, step=self.train_steps)
        self.logger.log_scalar('evaluation/episode_length', episode_length, step=self.train_steps)
        self.logger.log_scalar(f'evaluation/mean_reward_over_{self.mean_reward_window}_episodes',
                               np.mean(self.avg_return_pool), step=self.train_steps)

        if render:
            self.logger.log_video_from_frames("live-play", render_frames, fps=16, step=self.train_steps)

    def play(self, env: Environment) -> (float, int):
        """play.
       Play the environments using the current
       policy, without learning, for as many
       episodes as requested.
       """
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.EVAL)
        episodes = self.context.epochs
        render = self.context.render
        device = "cuda" if self.context.gpu and torch.cuda.is_available() else "cpu"
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

        mean_reward = np.mean(ep_rewards)
        mean_length = np.mean(ep_length)
        return f"Playing finished -- Mean reward {mean_reward:.2f}, mean episode length: {mean_length:.2f}", \
               mean_reward, mean_length

    def save(self, save_path: str) -> None:
        self.policy.save(save_path)

    def load(self, load_path: str) -> None:
        self.policy.load(load_path)
