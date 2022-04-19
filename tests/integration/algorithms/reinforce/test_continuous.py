import mock
import pytest
from relezoo.algorithms.reinforce.continuous import ReinforceContinuousPolicy, ReinforceContinuous
from relezoo.environments import GymWrapper
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES
from tests.utils.netpol import build_net


def build_policy(env: GymWrapper, learning_rate: float = 1e-2):
    in_shape = env.get_observation_space()[1]
    out_shape = env.get_action_space()[1]
    return ReinforceContinuousPolicy(
        build_net(in_shape, out_shape),
        learning_rate)


@pytest.mark.integration
class TestReinforceContinuousInt:
    @mock.patch("tensorboardX.SummaryWriter")
    def test_smoke_train_reinforce(self, mock_logger):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env)
        algo = ReinforceContinuous(policy=policy, logger=mock_logger)
        ctx = Context({
            "episodes": MAX_TEST_EPISODES,
            "render": False
        })
        algo.train(env, ctx)
        assert mock_logger.add_scalar.call_count == MAX_TEST_EPISODES * 3

    @mock.patch("tensorboardX.SummaryWriter")
    def test_smoke_play_reinforce(self, mock_logger):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env)
        algo = ReinforceContinuous(policy=policy, logger=mock_logger)
        ctx = Context({
            "episodes": MAX_TEST_EPISODES,
            "render": False
        })
        rewards, lengths = algo.play(env, ctx)
        assert isinstance(rewards, float)
        assert isinstance(lengths, float)
        assert rewards != 0.0
        assert lengths > 0.0
