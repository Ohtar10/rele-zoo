import mock
import pytest

from relezoo.algorithms.reinforce.discrete import ReinforceDiscrete, ReinforceDiscretePolicy
from relezoo.environments import GymWrapper
from relezoo.environments.base import Environment
from tests.utils.netpol import build_net


def build_policy(env: Environment, learning_rate: float = 1e-2):
    in_shape = env.get_observation_space()[0]
    out_shape = env.get_action_space()[0]
    return ReinforceDiscretePolicy(
        build_net(in_shape, out_shape),
        learning_rate)


@mock.patch("tensorboardX.SummaryWriter")
def test_smoke_train_reinforce(mock_logger):
    env = GymWrapper("CartPole-v0")
    policy = build_policy(env)
    algo = ReinforceDiscrete(env, policy, mock_logger)
    algo.train(epochs=5)
    assert mock_logger.add_scalar.call_count == 5 * 3  # 5 epochs * 3 metrics
    assert mock_logger.add_video.call_count == 2  # once in the beginning and once in the end


@mock.patch("tensorboardX.SummaryWriter")
def test_smoke_play_reinforce(mock_logger):
    env = GymWrapper("CartPole-v0")
    policy = build_policy(env)
    algo = ReinforceDiscrete(env, policy, mock_logger)
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
    env = GymWrapper(env_name)
    policy = build_policy(env)
    algo = ReinforceDiscrete(env, policy, mock_logger)
    algo.train(epochs=5)
    assert mock_logger.add_scalar.call_count == 5 * 3  # 5 epochs * 3 metrics
    assert mock_logger.add_video.call_count == 2  # once in the beginning and once in the end

