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

    def init(self, name: str, **kwargs):
        self.env = gym.make(name, **kwargs)
        self.env.reset()

    def step(self, action) -> Any:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def seed(self, seed: int):
        self.env.seed(seed)


class ParallelGym(Environment, ABC):
    """Multi Agent Parallel Gym environments.

    Using this class is equivalent to spawn multiple
    gym environment instances of the same type each
    with their own independent process. However, the class
    will automatically distribute actions and aggregate
    observations on each interaction. As consequence,
    the observation and action spaces reflect the number
    of parallel environments created.


    """
    def __init__(self, name: str, num_envs: int, **kwargs):
        """

        Parameters
        ----------
        name : str
            Name of the gym environment
        num_envs : int
            Number of parallel environments to manage
        kwargs : dict
            Gym environment options to be passed down

        """
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
            self.__observation_space = (self.num_envs, self.__local.observation_space.n)

        if isinstance(self.__local.action_space, gym.spaces.Box):
            self.__action_space = (self.num_envs,) + self.__local.action_space.shape
        elif isinstance(self.__local.action_space, gym.spaces.Discrete):
            self.__action_space = (self.num_envs, self.__local.action_space.n)

    def _build_envs(self) -> None:
        for _ in range(self.num_envs):
            self.__envs.append(RayGym.remote())
        for i in range(self.num_envs):
            self.__envs[i].init.remote(self.name, **self.params)

    def build_environment(self) -> gym.Env:
        pass

    def get_observation_space(self) -> Tuple[int]:
        return self.__observation_space

    def get_action_space(self) -> Tuple[int]:
        return self.__action_space

    def reset(self, idx: Optional[int] = None) -> Any:
        if idx is None:
            obs = [ray.get(e.reset.remote()) for e in self.__envs]
            return np.array(obs)
        else:
            obs = ray.get(self.__envs[idx].reset.remote())
            return np.expand_dims(obs, axis=0)

    def step(self, actions: Any) -> Any:
        """Take a step in every environment using the provided actions.

        Parameters
        ----------
        actions : Any
            It should be of shape ``get_action_space``, i.e., the actions
            to be sent to all the managed environments.

        Returns
        -------
        (obs, rewards, dones, infos) : Any
            Tuple with the observations, rewards, done signals and environment
            info for all the managed environments.

        """
        if isinstance(self.__local.action_space, gym.spaces.Discrete):
            actions = actions.squeeze().tolist()
        result = [ray.get(e.step.remote(a)) for a, e in zip(actions, self.__envs)]
        obs = np.array([r[0] for r in result])
        rewards = np.expand_dims(np.array([r[1] for r in result]), axis=1)
        dones = np.expand_dims(np.array([r[2] for r in result]), axis=1)
        infos = np.expand_dims(np.array([r[3] for r in result]), axis=1)
        return obs, rewards, dones, infos

    def render(self, mode: Optional[str] = None) -> Any:
        pass

    def seed(self, seed: int):
        for i, env in enumerate(self.__envs, start=1):
            env.seed.remote(seed + i)


