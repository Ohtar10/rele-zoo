from typing import Any, Optional
import numpy as np
import gym
from gym import Env

from relezoo.environments.base import Environment


class GymWrapper(Environment):
    """Open AI gym environment wrapper.

    This class wraps certain aspects
    of gym environments and acts as a
    proxy between ReleZoo and gym Wrappers
    using the :py:class:`relezoo.environments.base.Environment`
    class as a contract. The environments
    using this class are mono agent. For multi agent
    versions check :py:class:`relezoo.environments.parallel.ParallelGym`

    """

    def __init__(self, name: str, **kwargs):
        """

        Parameters
        ----------
        name : str
            Name of the gym environment
        kwargs : dict
            Gym environment options to be passed down

        """
        self.name = name
        self.params = kwargs
        # This is a local copy of the env only for
        # extracting metadata.
        self.__env: Env = gym.make(self.name)
        self.__observation_space = None
        self.__action_space = None
        self._build_metadata()

    def _build_metadata(self) -> None:
        if isinstance(self.__env.observation_space, gym.spaces.Box):
            self.__observation_space = (1,) + self.__env.observation_space.shape
        elif isinstance(self.__env.observation_space, gym.spaces.Discrete):
            self.__observation_space = (1, self.__env.observation_space.n)

        if isinstance(self.__env.action_space, gym.spaces.Box):
            self.__action_space = (1,) + self.__env.action_space.shape
        elif isinstance(self.__env.action_space, gym.spaces.Discrete):
            self.__action_space = (1, self.__env.action_space.n)

    def get_observation_space(self) -> Any:
        return self.__observation_space

    def get_action_space(self) -> Any:
        return self.__action_space

    def build_environment(self) -> gym.Env:
        return gym.make(self.name, **self.params)

    def reset(self, idx: Optional[int] = None) -> Any:
        # in a single gym env, idx is just ignored
        return np.expand_dims(self.__env.reset(), axis=0)

    def step(self, actions) -> Any:
        if isinstance(self.__env.action_space, gym.spaces.Box):
            obs, reward, done, info = self.__env.step(actions.reshape(1))
        elif isinstance(self.__env.action_space, gym.spaces.Discrete):
            obs, reward, done, info = self.__env.step(np.squeeze(actions))
        else:
            raise ValueError("action does not match with expected action space")
        obs = np.expand_dims(obs, axis=0)
        reward = np.expand_dims(reward, axis=0)
        done = np.expand_dims(done, axis=0)
        return obs, reward, done, info

    def render(self, mode: Optional[str] = None) -> Any:
        if mode is not None:
            return self.__env.render(mode)
        else:
            return self.__env.render()

    def seed(self, seed: int):
        self.__env.seed(seed)



