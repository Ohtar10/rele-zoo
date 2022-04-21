import mock
import pytest
from relezoo.algorithms.reinforce.continuous import ReinforceContinuousPolicy, ReinforceContinuous
from relezoo.environments import GymWrapper
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES
from tests.utils.netpol import build_net


def build_policy(env: GymWrapper, policy_class, learning_rate: float = 1e-2):
    in_shape = env.get_observation_space()[1]
    out_shape = env.get_action_space()[1]
    return policy_class(
        build_net(in_shape, out_shape),
        learning_rate)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("algo_class", "policy_class"),
    [
        (ReinforceContinuous, ReinforceContinuousPolicy)
    ]
)
class TestContinuousAlgorithmsIntegration:
    @mock.patch("tensorboardX.SummaryWriter")
    def test_smoke_train_reinforce(self, mock_logger, algo_class, policy_class):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy, logger=mock_logger)
        ctx = Context({
            "epochs": MAX_TEST_EPISODES,
            "render": False,
            "gpu": False
        })
        algo.train(env, ctx)
        assert mock_logger.add_scalar.call_count == MAX_TEST_EPISODES * 3

    @mock.patch("tensorboardX.SummaryWriter")
    def test_smoke_play_reinforce(self, mock_logger, algo_class, policy_class):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy, logger=mock_logger)
        ctx = Context({
            "epochs": MAX_TEST_EPISODES,
            "render": False,
            "gpu": False
        })
        rewards, lengths = algo.play(env, ctx)
        assert isinstance(rewards, float)
        assert isinstance(lengths, float)
        assert rewards != 0.0
        assert lengths > 0.0
