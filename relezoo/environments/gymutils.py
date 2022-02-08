from typing import Optional, Any

import gym
from gym import Env


class GymWrapper:
    """GymWrapper.
    This class wraps certain aspects
    of gym environments and acts as a
    builder and meta-data store.
    """
    def __init__(self, name: str, alias: Optional[str] = None):
        self.name = name
        self.alias = name if alias is None else alias
        # This is a local copy of the env only for
        # extracting metadata.
        self.__env: Env = gym.make(self.name)

    def build_env(self) -> gym.Env:
        return gym.make(self.name)

    def get_observation_space(self) -> Any:
        # TODO Not all gym environments report the obs space like this
        return self.__env.observation_space.shape

    def get_action_space(self) -> Any:
        # TODO Not all gym environments report the action space like this
        return self.__env.action_space


