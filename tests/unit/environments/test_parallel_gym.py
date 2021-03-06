import gym
import pytest
import numpy as np
import ray

from relezoo.environments.parallel import ParallelGym

parameters = [
    "CartPole-v0",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v2",
    "Pendulum-v1"
]


@pytest.fixture(autouse=True)
def run_around_tests():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(params=parameters)
def test_case(request):
    return request.param


@pytest.mark.unit
class TestParallelGym:

    def test_create_parallel_environments(self, test_case):
        num_envs = 5
        env_name = test_case
        l_env: gym.Env = gym.make(env_name)
        p_env = ParallelGym(env_name, num_envs)
        assert p_env.get_observation_space() == (num_envs,) + l_env.observation_space.shape
        if isinstance(l_env.action_space, gym.spaces.Box):
            assert p_env.get_action_space() == (num_envs,) + l_env.action_space.shape
        elif isinstance(l_env.action_space, gym.spaces.Discrete):
            assert p_env.get_action_space() == (num_envs, l_env.action_space.n)
        else:
            pytest.fail("Action space not matching")

        assert len(p_env._ParallelGym__envs) == num_envs

    def test_reset_parallel_environments(self, test_case):
        num_envs = 5
        env_name = test_case
        p_env = ParallelGym(env_name, num_envs)
        result = p_env.reset()
        assert result.shape == p_env.get_observation_space()

    def test_step_parallel_environments(self, test_case):
        num_envs = 5
        env_name = test_case
        l_env: gym.Env = gym.make(env_name)
        p_env = ParallelGym(env_name, num_envs)
        if isinstance(l_env.action_space, gym.spaces.Box):
            actions = np.zeros(p_env.get_action_space(), dtype=int)
        elif isinstance(l_env.action_space, gym.spaces.Discrete):
            actions = np.zeros((num_envs, 1), dtype=int)
        else:
            pytest.fail("Action space not matching")

        obs, rewards, dones, _ = p_env.step(actions)
        assert obs.shape == p_env.get_observation_space()
        assert rewards.shape == (num_envs, 1)
        assert dones.shape == (num_envs, 1)
