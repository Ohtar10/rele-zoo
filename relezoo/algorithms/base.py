from abc import ABC, abstractmethod

import torch

from relezoo.environments.base import Environment


class Algorithm(ABC):
    """Algorithm.

    Represents an RL algorithm to solve a task.
    """
    @abstractmethod
    def train(self, env: Environment, episodes: int, render: bool) -> None:
        pass

    @abstractmethod
    def play(self, env: Environment, episodes: int, render: bool) -> (float, int):
        pass

    @abstractmethod
    def save(self, save_path: str) -> None:
        pass

    @abstractmethod
    def load(self, save_path: str) -> None:
        pass


class Policy(ABC):
    """Policy.

    Represents a policy of an on-policy RL algorithm.
    """
    @abstractmethod
    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        pass

    @abstractmethod
    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: torch.Tensor):
        pass

    @abstractmethod
    def save(self, save_path):
        pass
