import mock
import pytest
from kink import di
from relezoo.algorithms.reinforce import Reinforce, ReinforceContinuousPolicy
from relezoo.environments import GymWrapper
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES
from tests.utils.netpol import build_net


@pytest.fixture(autouse=True)
def run_around_tests():
    di[Context] = Context({
        "epochs": MAX_TEST_EPISODES,
        "render": False,
        "gpu": False,
        "mean_reward_window": 100
    })
    di[Logging] = mock.MagicMock(Logging)
    yield
    di.clear_cache()


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
        (Reinforce, ReinforceContinuousPolicy)
    ]
)
class TestContinuousAlgorithmsIntegration:

    def test_smoke_train(self, algo_class, policy_class):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env, policy_class)
        mock_logger = di[Logging]
        algo = algo_class(policy=policy)
        algo.train(env)
        assert mock_logger.log_scalar.call_count == MAX_TEST_EPISODES * 3

    def test_smoke_play(self, algo_class, policy_class):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy)
        _, rewards, lengths = algo.play(env)
        assert isinstance(rewards, float)
        assert isinstance(lengths, float)
        assert rewards != 0.0
        assert lengths > 0.0
