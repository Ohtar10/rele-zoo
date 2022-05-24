import mock
import pytest
from kink import di
from numpy.polynomial import polynomial
import numpy as np
import torch
import random

from relezoo.algorithms.reinforce.discrete import ReinforceDiscretePolicy
from relezoo.algorithms.reinforce.core import Reinforce
from relezoo.algorithms.xentropy import CrossEntropyDiscretePolicy
from relezoo.algorithms.xentropy import CrossEntropyMethod
from relezoo.environments import GymWrapper
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES, get_test_context, TEST_SEED
from tests.utils.netpol import build_net


@pytest.fixture(autouse=True)
def run_around_tests():
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    di[Context] = get_test_context()
    di[Logging] = mock.MagicMock(Logging)
    yield
    di.clear_cache()


def build_policy(env: Environment, policy_class, learning_rate: float = 1e-2):
    in_shape = env.get_observation_space()[1]
    out_shape = env.get_action_space()[1]
    return policy_class(
        build_net(in_shape, out_shape),
        learning_rate)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("algo_class", "policy_class"),
    [
        (CrossEntropyMethod, CrossEntropyDiscretePolicy),
        (Reinforce, ReinforceDiscretePolicy)
    ]
)
class TestDiscreteAlgorithmsIntegration:

    def test_smoke_train(self, algo_class, policy_class):
        env = GymWrapper("CartPole-v0")
        mock_logger = di[Logging]
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy)
        algo.train(env, env)
        assert mock_logger.log_scalar.call_count == MAX_TEST_EPISODES * 6  # n epochs * 3 metrics (train + eval)

    def test_smoke_play(self, algo_class, policy_class):
        env = GymWrapper("CartPole-v0")
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy)
        _, rewards, lengths = algo.play(env)
        assert isinstance(rewards, float)
        assert isinstance(lengths, float)
        assert rewards > 0.0
        assert lengths > 0.0

    @pytest.mark.parametrize("env_name", [
        "CartPole-v0",
        "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1"
    ])
    def test_train_environments(self, env_name: str, algo_class, policy_class):
        env = GymWrapper(env_name)
        mock_logger = di[Logging]
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy)
        algo.train(env, env)
        assert mock_logger.log_scalar.call_count == MAX_TEST_EPISODES * 6  # n epochs * 3 metrics

    @pytest.mark.parametrize(("env_name", "delta", "overrides"), [
        ("CartPole-v1", 10.0, {
            ReinforceDiscretePolicy: {"batch_size": 4096},
            CrossEntropyDiscretePolicy: {"batch_size": 4096}
        })
    ])
    def test_improvement(self, env_name: str, delta: float, overrides, algo_class, policy_class):
        env = GymWrapper(env_name)
        env.seed(TEST_SEED)
        context = di[Context]
        context.epochs = 10
        policy = build_policy(env, policy_class)
        algo = algo_class(policy=policy)

        if policy_class in overrides:
            policy_overrides = overrides[policy_class]
            for k, v in policy_overrides.items():
                policy.__setattr__(k, v)

        algo.train(env, env)
        returns = algo.avg_return_pool
        x = list(range(len(returns)))
        slope = polynomial.polyfit(x, returns, 1)[0]
        assert slope >= delta





