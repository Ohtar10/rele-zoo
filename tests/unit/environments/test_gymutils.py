import gym
import pytest

from relezoo.environments import GymWrapper


@pytest.mark.parametrize("env_name", [
    "CartPole-v0",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v2"
])
def test_build_discrete_action_envs(env_name):
    builder = GymWrapper(env_name)
    env = builder.build_env()
    assert isinstance(env, gym.Env)


@pytest.mark.parametrize("env_name", [
    "Pendulum-v1",
    "BipedalWalker-v3",
    "MountainCarContinuous-v0"
])
def test_build_continuous_action_envs(env_name):
    builder = GymWrapper(env_name)
    env = builder.build_env()
    assert isinstance(env, gym.Env)


@pytest.mark.parametrize(("env_name", "params"), [
    ("Pendulum-v1", {"g": 9.81}),
    ("Pendulum-v1", {"g": 10.1}),
    ("Pendulum-v1", {"g": 5.0}),
])
def test_build_environments_with_parameters(env_name, params):
    builder = GymWrapper(env_name, **params)
    env = builder.build_env()
    assert isinstance(env, gym.Env)
    for k, v in params.items():
        assert hasattr(env.env, k)
        assert getattr(env.env, k) == v

