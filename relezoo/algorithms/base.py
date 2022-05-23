from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Optional, Any, Tuple

import os
import numpy as np
import torch
from kink import inject
from numpy import ndarray
from tqdm import tqdm

from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.network import NetworkMode
from relezoo.utils.structure import Context


@inject
class Policy(ABC):
    """Represents a policy of an on-policy RL algorithm.

    The Policy class has two major responsibilities:
    # Take actions given observations.
    # Learn from and update the current policy given
    some batch information (observations, actions, and weights)

    The policy then can be used detached from the training
    algorithm which is usually not necessary when using
    the policy live.

    Attributes
    ----------

    context: Context
        The experiment context with general hyper-parameters.

    """

    def __init__(self, context: Context = None):
        """

        Parameters
        ----------
        context: Context
            The experiment context with general hyper-parameters.
        """
        self.context = context
        self.nets = {}
        self.device = "cpu"

    @abstractmethod
    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        """Take action given the received observations.

        It should be expected to receive the observations
        in the shape (#agents, observation shape) such that
        this policy can take actions from multiple agent
        observations. Similarly, it will return a tensor
        of shape (#agents, action shape).

        Parameters
        ----------
        obs : torch.Tensor
            Observations Tensor. Expected shape (#agents, observation shape)

        Returns
        -------
        actions: torch.Tensor
            Actions Tensor. Expected shape (#agents, action shape)

        """
        pass

    @abstractmethod
    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        """Learn and update the current policy from the given data.

        This method execute the custom algorithm learning
        routine from the given observations.

        Parameters
        ----------
        batch_obs: torch.Tensor
            Tensor with all the observations of a training batch.
        batch_actions: torch.Tensor
            Tensor with all the actions corresponding to the observations.
        batch_weights: torch.Tensor
            Tensor with all weight values for each obs,act pair.

        """
        pass

    def save(self, save_path: str):
        """Saves the current policy elements.

        Parameters
        ----------
        save_path : str
            Directory where to save the policy elements.

        """
        for key, value in self.nets.items():
            path = os.path.join(save_path, value)
            torch.save(self.__getattribute__(key), path)

    def load(self, load_path: str):
        """Update the current networks with the checkpoints in the provided path.

        Parameters
        ----------
        load_path : str
            Directory where to load the checkpoints from.

        """
        device = "cuda" if self.context and self.context.gpu and torch.cuda.is_available() else "cpu"
        for key, value in self.nets.items():
            path = os.path.join(load_path, value)
            self.__setattr__(key, torch.load(path, map_location=torch.device(device)))

    def set_mode(self, mode: NetworkMode):
        """Utility to set the models/networks in a particular mode.

        Parameters
        ----------
        mode : NetworkMode
            Check :py:class:relezoo.utils.network.NetworkMode

        """
        for key in self.nets.keys():
            if mode == NetworkMode.TRAIN:
                self.__getattribute__(key).train()
            else:
                self.__getattribute__(key).eval()

    def to(self, device: str):
        """Moves the network computations to the given device.

        Parameters
        ----------
        device : str
            Device where to move the network to.

        """
        self.device = device
        for key in self.nets.keys():
            self.__setattr__(key, self.__getattribute__(key).to(device))


class Algorithm(ABC):
    """Represents an RL algorithm to solve a task.

    Attributes
    ----------
    context : Context
        General context parameters. (Injectable)
    logging : Logging
        Logging mechanism. (Injectable)
    batch_size : int
        Number of rollouts to collect before train step
    policy : Policy
        The Policy to use and train the algorithm on

    """

    def __init__(self,
                 context: Context,
                 logging: Logging,
                 batch_size: int,
                 policy: Optional[Policy] = None):
        """

        Parameters
        ----------
        context : Context
            The experiment context with global hyper-parameters.
        logging : Logging
            The logging mechanism to use.
        batch_size : int
            Number of elements per batch per training epoch.
        policy : Policy
            The policy instance to use.
        """
        self.context = context
        self.logger = logging
        self.policy = policy
        self.batch_size = batch_size
        self.mean_reward_window = context.mean_reward_window
        self.avg_return_pool = deque(maxlen=self.mean_reward_window)
        self.train_steps = max(0, context.start_at_step - 1)

    def train(self, env: Environment, eval_env: Optional[Environment] = None) -> Any:
        """The main training loop for the algorithm.

        Controls the main training loop making as
        many calls as needed to `train_epoch`.

        Parameters
        ----------
        env : Environment
            The training environment
        eval_env : Environment
            The evaluation environment
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
                self.logger.flush()
            self.logger.close()

        mean_reward = np.mean(self.avg_return_pool)
        return f"Training finished -- Mean reward over {self.mean_reward_window} epochs: " \
               f"{mean_reward:.2f}", mean_reward

    @abstractmethod
    def train_epoch(self, env: Environment, batch_size: int):
        """Train one epoch as per Algorithm implementation

        Parameters
        ----------
        env : Environment
            Training Environment.
        batch_size : int
            The batch size for this training epoch.

        """
        pass

    def evaluate(self, env: Environment, render: bool = False):
        """Evaluate the current policy against the provided environment

        Parameters
        ----------
        env : Environment
            The environment the policy will be evaluated on.
        render : bool
            Determines if the evaluation rollout should be rendered

        """
        render_frames = []
        episode_return = 0
        episode_length = 0
        obs = env.reset()
        while True:
            if render:
                render_frames.append(env.render(mode='rgb_array'))

            action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            episode_return += reward[0]
            episode_length += 1
            if done:
                break

        self.avg_return_pool.append(episode_return)
        self.logger.log_scalar("evaluation/return", episode_return, step=self.train_steps)
        self.logger.log_scalar("evaluation/num_samples", self.train_steps * self.batch_size, step=self.train_steps)
        self.logger.log_scalar('evaluation/episode_length', episode_length, step=self.train_steps)
        self.logger.log_scalar(f'evaluation/mean_reward_over_{self.mean_reward_window}_episodes',
                               np.mean(self.avg_return_pool), step=self.train_steps)
        if render:
            self.logger.log_video_from_frames(
                "live-play", render_frames, fps=self.context.render_fps, step=self.train_steps
            )
            self.logger.log_table_row(
                "play_progress",
                OrderedDict({
                    "step": "noop",
                    "video": f"video_file(logging-video-{self.train_steps}.mp4)",
                    "episode_reward": "noop",
                    "episode_length": "noop",
                    f'mean_reward_over_{self.mean_reward_window}_episodes': "noop"}),
                [
                    self.train_steps,
                    render_frames,
                    episode_return,
                    episode_length,
                    np.mean(self.avg_return_pool)
                ]
            )

    def play(self, env: Environment) -> Tuple[str, ndarray, ndarray]:
        """Use the current policy to play in the provided environment.

        Parameters
        ----------
        env : Environment
            The environment for rollouts

        Returns
        -------
        msg : str
            Result message with mean reward and mean length result.
        mean_reward : float
            Mean reward obtained among the total rollouts.
        mean_length : float
            Mean episode length among the total rollouts.
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
                render_frames = []
                obs = env.reset()
                ep_length = 1
                ep_reward = 0
                while True:
                    if render:
                        render_frames.append(env.render(mode='rgb_array'))

                    action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
                    obs, reward, done, _ = env.step(action)
                    ep_reward += reward
                    if done:
                        break
                    ep_length += 1
                ep_lengths.append(ep_length)
                ep_rewards.append(ep_reward)
                self.logger.log_scalar("play/episode_reward", ep_reward, step=i)
                self.logger.log_scalar("play/episode_length", ep_length, step=i)
                if render:
                    self.logger.log_video_from_frames(
                        "live-play", render_frames, fps=self.context.render_fps, step=i
                    )
                progress.set_postfix({
                    "Avg. reward": f"{np.mean(ep_rewards):.2f}",
                    "Avg. ep length": f"{np.mean(ep_length):.2f}"
                })
                self.logger.flush()
                progress.update()
            self.logger.log_scalar("play/mean_episode_reward", np.mean(ep_rewards))
            self.logger.log_scalar("play/mean_episode_length", np.mean(ep_length))
        self.logger.close()
        mean_reward = np.mean(ep_rewards)
        mean_length = np.mean(ep_length)
        return f"Playing finished -- Mean reward {mean_reward:.2f}, mean episode length: {mean_length:.2f}", \
               mean_reward, mean_length

    def _log(self, batch_loss, batch_returns, batch_lens):
        """General metric logging.

        Parameters
        ----------
        batch_loss : float
            Loss obtained during one training batch.
        batch_returns : List[float]
            List of rewards per batch step/episode.
        batch_lens: List[int]
            List of episode lengths for a batch.

        """
        if self.logger is not None:
            self.logger.log_scalar('training/loss', batch_loss, self.train_steps)
            self.logger.log_scalar('training/return', np.mean(batch_returns), self.train_steps)
            self.logger.log_scalar('training/mean_episode_length', np.mean(batch_lens), self.train_steps)

    def save(self, save_path: str) -> None:
        """Forward to the Policy's saving routine.

        Parameters
        ----------
        save_path : str
            Directory where to store the policy checkpoints.

        """
        self.policy.save(save_path)

    def load(self, load_path: str) -> None:
        """Forward to the Policy's loading routine.

        Parameters
        ----------
        load_path : str
            Directory where to load from the policy checkpoints.

        """
        self.policy.load(load_path)
