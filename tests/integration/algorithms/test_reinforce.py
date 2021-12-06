import pytest
import gym
import torch.nn as nn
import mock
from relezoo.algorithms.reinforce import Reinforce


@mock.patch("tensorboardX.SummaryWriter")
def test_smoke_train_reinforce(mock_logger):
    env = gym.make("CartPole-v0")
    net = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 32),
        nn.Tanh(),
        nn.Linear(32, env.action_space.n)
    )
    algo = Reinforce(env, net, mock_logger, epochs=5, render=True)
    algo.train()
    assert mock_logger.add_scalar.call_count == 5 * 3  # 5 epochs * 3 metrics
    assert mock_logger.add_video.call_count == 2  # once in the beginning and once in the end


@pytest.mark.parametrize("env", [
    "CartPole-v0"
])
def test_train_reinforce_environments(env):
    pytest.fail("Not yet implemented")
