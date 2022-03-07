import gym
import mock

from relezoo.algorithms.reinforce.continuous import ReinforceContinuousPolicy, ReinforceContinuous
from relezoo.environments import GymWrapper
from tests.utils.netpol import build_net


def build_policy(env: GymWrapper, learning_rate: float = 1e-2):
    in_shape = env.get_observation_space()[0]
    out_shape = env.get_action_space()[0]
    return ReinforceContinuousPolicy(
        build_net(in_shape, out_shape),
        learning_rate)


@mock.patch("tensorboardX.SummaryWriter")
def test_smoke_train_reinforce(mock_logger):
    env = GymWrapper("Pendulum-v1")
    policy = build_policy(env)
    algo = ReinforceContinuous(env, policy=policy, logger=mock_logger)
    algo.train(episodes=5)
    assert mock_logger.add_scalar.call_count == 5 * 3
    assert mock_logger.add_video.call_count == 2

