import abc
from abc import abstractmethod
from typing import Any, Optional, Union, Tuple
import numpy as np


class Environment(abc.ABC):
    """Represents a RL task environment.

    This is an abstract base class with the general
    functionality contract all the environments should
    provide to ensure compatibility with the Algorithms.

    """

    @abstractmethod
    def build_environment(self) -> Any:
        pass

    @abstractmethod
    def get_observation_space(self) -> Union[np.ndarray, Tuple[int]]:
        """Obtain the observation space of the environment.

        **Note:** You should expect the first entry in the returned
        shape to be the number of agents the environment supports.
        If the environment only support one agent, you should expect
        a value of 1.

        Example: a shape (1, 4) means **one** agent with an observation
        space of 4 dimensions.

        Returns
        -------
        Union[np.ndarray, Tuple[int]]
           Shape either as a numpy array or as a python tuple.

        """
        pass

    @abstractmethod
    def get_action_space(self) -> Union[np.ndarray, Tuple[int]]:
        """Obtain the action space of the agent in the environment.

        **Note:** You should expect the first entry in the returned
        shape to be the number of agents the environment supports.
        If the environment only supports one agent, you should expect
        a value of 1.

        Example: a shape (1, 2) means **one** agent with an action
        space of 2 dimensions.

        Returns
        -------
        Union[np.ndarray, Tuple[int]]
           Shape either as a numpy array or as a python tuple.

        """
        pass

    @abstractmethod
    def reset(self, idx: Optional[int] = None) -> Any:
        """Resets the environment to its initial state.

        Parameters
        ----------
        idx : Optional[int]
            index of the agent to reset the environment. Only
            useful on multi agent environments, otherwise ignored.

        Returns
        -------
        state : Any
            The initial environment state/observation

        """
        pass

    @abstractmethod
    def step(self, actions: Any) -> Any:
        """Take a step in the environment using the provided action.

        **Note:** On multi agent environments, the provided actions
        should be of shape (# agents, action space). i.e., it should
        be the same shape as ``get_action_space``.

        Parameters
        ----------
        actions : Any
            The action array for every agent in the environment.

        Returns
        -------
        (state, reward, done, info): Tuple[Any]
            A tuple with the next state and reward produced
            by the interaction, if the environment is finished
            and info about the environment.


        """
        pass

    @abstractmethod
    def render(self, mode: Optional[str] = None) -> Any:
        """Returns a visual representation of the current state.

        Parameters
        ----------
        mode : Optional[str]
            Either `human`, `array`, or environment default

        Returns
        -------
        render : Any
            Either an image or the numpy representation.

        """
        pass

    @abstractmethod
    def seed(self, seed: int):
        """Set the environment seed for reproducibility.

        Parameters
        ----------
        seed : int
            Seed value to pass to the environment

        Returns
        -------
        Nothing

        """
        pass

