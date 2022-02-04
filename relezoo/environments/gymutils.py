from typing import Optional, Any

import gym
from gym import Env


class GymWrapper:

    def __init__(self, name: str, alias: Optional[str] = None):
        self.name = name
        self.alias = name if alias is None else alias
        self.__env: Env = gym.make(self.name)

    def build_env(self) -> gym.Env:
        return gym.make(self.name)

    def get_observation_space(self) -> Any:
        return self.__env.observation_space.shape

    def get_action_space(self) -> Any:
        return self.__env.action_space


