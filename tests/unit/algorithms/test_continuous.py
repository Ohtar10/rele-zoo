import mock
import numpy as np
import pytest
import torch
import torch.nn as nn
from kink import di

from relezoo.algorithms.base import Policy
from relezoo.algorithms.reinforce import Reinforce, ReinforceContinuousPolicy
from relezoo.algorithms.xentropy import CrossEntropyMethod, CrossEntropyContinuousPolicy
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES, get_test_context


@pytest.fixture(autouse=True)
def run_around_tests():
    di[Context] = get_test_context()
    di[Logging] = mock.MagicMock(Logging)
    yield
    di.clear_cache()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("algo_class", "policy_class"),
    [
        (Reinforce, ReinforceContinuousPolicy),
        (CrossEntropyMethod, CrossEntropyContinuousPolicy)
    ]
)
class TestContinuousAlgorithms:

    @mock.patch("gym.Env")
    def test_train(self, mock_env, algo_class, policy_class):
        mock_policy = mock.MagicMock(policy_class)
        mock_logger = di[Logging]
        algo = algo_class(policy=mock_policy)
        algo.train_epoch = mock.MagicMock(return_value=(0.1, np.array([1, 2]), np.array([1, 2])))
        algo.train(mock_env)
        assert algo.train_epoch.call_count == MAX_TEST_EPISODES
        assert mock_logger.flush.call_count == MAX_TEST_EPISODES
        mock_logger.close.assert_called_once()

    def test_save_agent(self, algo_class,  policy_class):
        mock_policy = mock.MagicMock(policy_class)
        algo = algo_class(policy=mock_policy)
        save_path = "save-path"
        algo.save(save_path)
        mock_policy.save.called_once_with(save_path)

    @mock.patch("torch.nn.Module")
    def test_save_policy(self, mock_net, algo_class, policy_class):
        with mock.patch(".".join([Policy.__module__, "torch"])) as mocked_torch:
            dummy_parameters = [nn.Parameter(torch.tensor([1., 2.]))]
            mock_net.parameters.return_value = dummy_parameters
            policy = policy_class(mock_net)
            save_path = "save-path"
            policy.save(save_path)
            mocked_torch.save.called_once_with(mock_net)

    @mock.patch("torch.nn.Module")
    def test_load_agent(self, mock_net, algo_class, policy_class):
        with mock.patch(".".join([Policy.__module__, "torch"])) as mocked_torch:
            dummy_parameters = [nn.Parameter(torch.tensor([1., 2.]))]
            mock_net.parameters.return_value = dummy_parameters
            mocked_torch.load.return_value = mock_net
            policy = policy_class(mock_net)
            algo = algo_class(policy=policy)
            load_path = "load-path"
            algo.load(load_path)
            assert algo.policy is not None
            mocked_torch.load.called_once_with(load_path)

    @mock.patch("torch.nn.Module")
    @mock.patch("gym.Env")
    def test_play(self, mock_env, mock_net, algo_class, policy_class):
        with mock.patch(".".join([policy_class.__module__, "torch", "from_numpy"])) as mock_from_numpy:
            dummy_parameters = [nn.Parameter(torch.tensor([1., 2.]))]
            mock_net.parameters.return_value = dummy_parameters
            policy = policy_class(mock_net)
            algo = algo_class(policy=policy)
            policy.act = mock.MagicMock(side_effect=torch.Tensor([0, 1, 1, 0]))
            steps = [
                (np.array([1, 1, 1]), 1, False, None),
                (np.array([1, 0, 1]), 2, False, None),
                (np.array([1, 1, 0]), 3, False, None),
                (np.array([1, 0, 0]), 4, True, None),
            ]
            mock_env.step.side_effect = steps

            mock_from_numpy.side_effect = [s[0] for s in steps]

            ctx = di[Context]
            ctx.epochs = 1

            _, avg_reward, avg_ep_length = algo.play(mock_env)
            mock_env.reset.assert_called_once()
            assert policy.act.call_count == 4
            assert mock_env.step.call_count == 4
            assert avg_reward == 10
            assert avg_ep_length == 4

    @mock.patch("gym.Env")
    def test_play_no_policy_should_fail(self, mock_env, algo_class, policy_class):
        algo = algo_class()
        with pytest.raises(AssertionError) as e:
            algo.play(mock_env)
            assert e.value == "The policy is not defined."

    @mock.patch("gym.Env")
    def test_train_no_policy_should_fail(self, mock_env, algo_class, policy_class):
        algo = algo_class()
        ctx = Context({
            "epochs": MAX_TEST_EPISODES,
            "render": False,
            "gpu": False
        })
        with pytest.raises(AssertionError) as e:
            algo.train(mock_env, ctx)
            assert e.value == "The policy is not defined."
