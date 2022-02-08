from abc import ABC, abstractmethod

import torch


class Algorithm(ABC):
    """Algorithm.

    Represents an RL algorithm to solve a task.
    """
    @abstractmethod
    def train(self, episodes: int, batch_size: int, render: bool) -> None:
        pass

    @abstractmethod
    def play(self, episodes: int, render: bool) -> (float, int):
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
