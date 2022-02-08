import pytest
import gym
import torch.nn as nn
import mock
from relezoo.algorithms.reinforce import Reinforce, ReinforcePolicy


def build_net(env: gym.Env):
    return nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 32),
        nn.Tanh(),
        nn.Linear(32, env.action_space.n)
    )


def build_policy(env: gym.Env, learning_rate: float = 1e-2):
    return ReinforcePolicy(build_net(env), learning_rate)


@mock.patch("tensorboardX.SummaryWriter")
def test_smoke_train_reinforce(mock_logger):
    env = gym.make("CartPole-v0")
    policy = build_policy(env)
    algo = Reinforce(env, policy, mock_logger)
    algo.train(epochs=5)
    assert mock_logger.add_scalar.call_count == 5 * 3  # 5 epochs * 3 metrics
    assert mock_logger.add_video.call_count == 2  # once in the beginning and once in the end


@mock.patch("tensorboardX.SummaryWriter")
def test_smoke_play_reinforce(mock_logger):
    env = gym.make("CartPole-v0")
    policy = build_policy(env)
    algo = Reinforce(env, policy, mock_logger)
    algo.train(epochs=5)
    assert mock_logger.add_scalar.call_count == 5 * 3  # 5 epochs * 3 metrics
    assert mock_logger.add_video.call_count == 2  # once in the beginning and once in the end


@mock.patch("tensorboardX.SummaryWriter")
@pytest.mark.parametrize("env_name", [
    "CartPole-v0",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1"
])
def test_train_reinforce_environments(mock_logger, env_name: str):
    env = gym.make(env_name)
    policy = build_policy(env)
    algo = Reinforce(env, policy, mock_logger)
    algo.train(epochs=5)
    assert mock_logger.add_scalar.call_count == 5 * 3  # 5 epochs * 3 metrics
    assert mock_logger.add_video.call_count == 2  # once in the beginning and once in the end

