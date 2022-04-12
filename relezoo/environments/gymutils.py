from typing import Any, Optional

import gym
from gym import Env
from gym.spaces import Box

from relezoo.environments.base import Environment


class GymWrapper(Environment):
    """GymWrapper.
    This class wraps certain aspects
    of gym environments and acts as a
    builder and meta-data store.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        # This is a local copy of the env only for
        # extracting metadata.
        self.__env: Env = gym.make(self.name)

    def get_observation_space(self) -> Any:
        return self.__env.observation_space.shape

    def get_action_space(self) -> Any:
        if isinstance(self.__env.action_space, Box):
            return self.__env.action_space.shape
        else:
            return [self.__env.action_space.n]

    def build_environment(self) -> gym.Env:
        return gym.make(self.name, **self.params)

    def reset(self) -> Any:
        return self.__env.reset()

    def step(self, actions) -> Any:
        return self.__env.step(actions)

    def render(self, mode: Optional[str] = None) -> Any:
        if mode is not None:
            return self.__env.render(mode)
        else:
            return self.__env.render()


