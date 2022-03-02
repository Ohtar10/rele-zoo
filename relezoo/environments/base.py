import abc
from abc import abstractmethod
from typing import Any, Optional
import numpy as np


class Environment(abc.ABC):

    @abstractmethod
    def build_environment(self) -> Any:
        pass

    @abstractmethod
    def get_observation_space(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_action_space(self) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, actions: Any) -> Any:
        pass

    @abstractmethod
    def render(self, mode: Optional[str] = None) -> Any:
        pass
