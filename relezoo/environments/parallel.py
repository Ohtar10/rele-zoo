from abc import ABC
from typing import Any, Optional, Tuple
import ray
import gym
import numpy as np

from relezoo.environments.base import Environment


@ray.remote
class RayGym(object):
    def __init__(self):
        self.env = None

    def init(self, name: str):
        self.env = gym.make(name)
        self.env.reset()

    def step(self, action) -> Any:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        return self.env.reset()


class ParallelGym(Environment, ABC):
    def __init__(self, name: str, num_envs: int, **kwargs):
        self.name = name
        self.num_envs = num_envs
        self.params = kwargs
        self.__observation_space = None
        self.__action_space = None
        self.__local: gym.Env = gym.make(name)
        self.__envs = []
        self._build_metadata()
        self._build_envs()

    def _build_metadata(self) -> None:
        if isinstance(self.__local.observation_space, gym.spaces.Box):
            self.__observation_space = (self.num_envs,) + self.__local.observation_space.shape
        elif isinstance(self.__local.observation_space, gym.spaces.Discrete):
            self.__observation_space = (self.num_envs, 1)

        if isinstance(self.__local.action_space, gym.spaces.Box):
            self.__action_space = (self.num_envs,) + self.__local.action_space.shape
        elif isinstance(self.__local.action_space, gym.spaces.Discrete):
            self.__action_space = (self.num_envs, 1)

    def _build_envs(self) -> None:
        for _ in range(self.num_envs):
            self.__envs.append(RayGym.remote())
        for i in range(self.num_envs):
            self.__envs[i].init.remote(self.name)

    def build_environment(self) -> gym.Env:
        pass

    def get_observation_space(self) -> Tuple[int]:
        return self.__observation_space

    def get_action_space(self) -> Tuple[int]:
        return self.__action_space

    def reset(self) -> Any:
        obs = [ray.get(e.reset.remote()) for e in self.__envs]
        return np.array(obs)

    def step(self, actions: Any) -> Any:
        if isinstance(self.__local.action_space, gym.spaces.Discrete):
            actions = actions.squeeze().tolist()
        result = [ray.get(e.step.remote(a)) for a, e in zip(actions, self.__envs)]
        obs = np.array([r[0] for r in result])
        rewards = np.expand_dims(np.array([r[1] for r in result]), axis=1)
        dones = np.expand_dims(np.array([r[2] for r in result]), axis=1)
        return obs, rewards, dones

    def render(self, mode: Optional[str] = None) -> Any:
        pass
