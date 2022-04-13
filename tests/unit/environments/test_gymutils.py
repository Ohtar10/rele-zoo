import gym
import pytest

from relezoo.environments import GymWrapper


@pytest.mark.unit
class TestGymWrapper:

    @pytest.mark.parametrize("env_name", [
        "CartPole-v0",
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "LunarLander-v2"
    ])
    def test_build_discrete_action_envs(self, env_name):
        env = GymWrapper(env_name)
        gym_env = env.build_environment()
        assert isinstance(gym_env, gym.Env)
        TestGymWrapper.assert_action_space(gym_env, env)

    @pytest.mark.parametrize("env_name", [
        "Pendulum-v1",
        "BipedalWalker-v3",
        "MountainCarContinuous-v0"
    ])
    def test_build_continuous_action_envs(self, env_name):
        env = GymWrapper(env_name)
        gym_env = env.build_environment()
        assert isinstance(gym_env, gym.Env)
        TestGymWrapper.assert_action_space(gym_env, env)

    @pytest.mark.parametrize(("env_name", "params"), [
        ("Pendulum-v1", {"g": 9.81}),
        ("Pendulum-v1", {"g": 10.1}),
        ("Pendulum-v1", {"g": 5.0}),
    ])
    def test_build_environments_with_parameters(self, env_name, params):
        builder = GymWrapper(env_name, **params)
        env = builder.build_environment()
        assert isinstance(env, gym.Env)
        for k, v in params.items():
            assert hasattr(env.env, k)
            assert getattr(env.env, k) == v

    @staticmethod
    def assert_observation_space(gym_env: gym.Env, env: GymWrapper):
        if isinstance(gym_env.observation_space, gym.spaces.Box):
            assert env.get_observation_space() == (1,) + gym_env.observation_space.shape
        elif isinstance(gym_env.observation_space, gym.spaces.Discrete):
            assert env.get_observation_space() == (1, 1)

    @staticmethod
    def assert_action_space(gym_env: gym.Env, env: GymWrapper):
        if isinstance(gym_env.action_space, gym.spaces.Box):
            assert env.get_action_space() == (1,) + gym_env.action_space.shape
        elif isinstance(gym_env.action_space, gym.spaces.Discrete):
            assert env.get_action_space() == (1, 1)
