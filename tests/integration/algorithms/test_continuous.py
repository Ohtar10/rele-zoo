import mock
import pytest
from kink import di
import numpy as np
import torch
from numpy.polynomial import polynomial
from relezoo.algorithms.reinforce import Reinforce, ReinforceContinuousPolicy
from relezoo.algorithms.xentropy import CrossEntropyMethod, CrossEntropyContinuousPolicy
from relezoo.environments import GymWrapper
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES, get_test_context, TEST_SEED
from tests.utils.netpol import build_net


@pytest.fixture(autouse=True)
def run_around_tests():
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    di[Context] = get_test_context()
    di[Logging] = mock.MagicMock(Logging)
    yield
    di.clear_cache()


def build_policy(env: GymWrapper, policy_class, learning_rate: float = 1e-2, net_type: str = "simple"):
    in_shape = env.get_observation_space()[1]
    out_shape = env.get_action_space()[1]
    return policy_class(
        build_net(in_shape, out_shape, net_type),
        learning_rate)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("algo_class", "policy_class", "net_type"),
    [
        (Reinforce, ReinforceContinuousPolicy, "musigma"),
        (CrossEntropyMethod, CrossEntropyContinuousPolicy, "musigma")
    ]
)
class TestContinuousAlgorithmsIntegration:

    def test_smoke_train(self, algo_class, policy_class, net_type):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env, policy_class, net_type=net_type)
        mock_logger = di[Logging]
        algo = algo_class(policy=policy)
        algo.train(env)
        assert mock_logger.log_scalar.call_count == MAX_TEST_EPISODES * 3

    def test_smoke_play(self, algo_class, policy_class, net_type):
        env = GymWrapper("Pendulum-v1")
        policy = build_policy(env, policy_class, net_type=net_type)
        algo = algo_class(policy=policy)
        _, rewards, lengths = algo.play(env)
        assert isinstance(rewards, float)
        assert isinstance(lengths, float)
        assert rewards != 0.0
        assert lengths > 0.0

    @pytest.mark.parametrize(("env_name", "delta"), [
        ("Pendulum-v1", 2.0)
    ])
    def test_improvement(self, env_name: str, delta: float, algo_class, policy_class, net_type):
        pytest.skip("Pending good algorithm/environment implementation")
        env = GymWrapper(env_name)
        policy = build_policy(env, policy_class, net_type=net_type)
        algo = algo_class(policy=policy)
        algo.train(env, env)
        returns = algo.avg_return_pool
        x = list(range(len(returns)))
        slope = polynomial.polyfit(x, returns, 1)[-1]
        assert slope >= delta
