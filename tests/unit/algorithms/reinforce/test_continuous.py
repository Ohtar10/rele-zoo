import mock
import numpy as np
import pytest
import torch
import torch.nn as nn

from relezoo.algorithms.reinforce.continuous import ReinforceContinuous, ReinforceContinuousPolicy
from relezoo.utils.structure import Context
from tests.utils.common import MAX_TEST_EPISODES


@pytest.mark.unit
class TestReinforceContinuous:
    @mock.patch("tensorboardX.SummaryWriter")
    @mock.patch("gym.Env")
    @mock.patch("relezoo.algorithms.reinforce.continuous.ReinforceContinuousPolicy")
    def test_reinforce_train(self, mock_policy, mock_env, mock_logger):
        algo = ReinforceContinuous(policy=mock_policy, logger=mock_logger)
        algo._train_epoch = mock.MagicMock(return_value=(0.1, np.array([1, 2]), np.array([1, 2])))
        ctx = Context({
            "episodes": MAX_TEST_EPISODES,
            "render": False
        })
        algo.train(mock_env, ctx)
        assert algo._train_epoch.call_count == MAX_TEST_EPISODES
        assert mock_logger.flush.call_count == MAX_TEST_EPISODES
        mock_logger.close.assert_called_once()

    @mock.patch("tensorboardX.SummaryWriter")
    @mock.patch("relezoo.algorithms.reinforce.continuous.ReinforceContinuousPolicy")
    def test_save_agent(self, mock_policy, mock_logger):
        algo = ReinforceContinuous(mock_policy, mock_logger)
        save_path = "save-path"
        algo.save(save_path)
        mock_policy.save.called_once_with(save_path)

    @mock.patch("torch.nn.Module")
    @mock.patch("relezoo.algorithms.reinforce.continuous.torch")
    def test_save_policy(self, mocked_torch, mock_net):
        dummy_parameters = [nn.Parameter(torch.tensor([1., 2.]))]
        mock_net.parameters.return_value = dummy_parameters
        policy = ReinforceContinuousPolicy(mock_net)
        save_path = "save-path"
        policy.save(save_path)
        mocked_torch.save.called_once_with(mock_net)

    @mock.patch("torch.nn.Module")
    @mock.patch("relezoo.algorithms.reinforce.continuous.torch")
    def test_load_agent(self, mocked_torch, mock_net):
        dummy_parameters = [nn.Parameter(torch.tensor([1., 2.]))]
        mock_net.parameters.return_value = dummy_parameters
        mocked_torch.load.return_value = mock_net
        algo = ReinforceContinuous()
        load_path = "load-path"
        algo.load(load_path)
        assert algo.policy is not None
        mocked_torch.load.called_once_with(load_path)

    @mock.patch("relezoo.algorithms.reinforce.discrete.torch.from_numpy")
    @mock.patch("torch.nn.Module")
    @mock.patch("gym.Env")
    def test_play(self, mock_env, mock_net, mock_from_numpy):
        dummy_parameters = [nn.Parameter(torch.tensor([1., 2.]))]
        mock_net.parameters.return_value = dummy_parameters
        policy = ReinforceContinuousPolicy(mock_net)
        algo = ReinforceContinuous(policy)
        policy.act = mock.MagicMock(side_effect=torch.Tensor([0, 1, 1, 0]))
        steps = [
            (np.array([1, 1, 1]), 1, False, None),
            (np.array([1, 0, 1]), 2, False, None),
            (np.array([1, 1, 0]), 3, False, None),
            (np.array([1, 0, 0]), 4, True, None),
        ]
        mock_env.step.side_effect = steps

        mock_from_numpy.side_effect = [s[0] for s in steps]

        ctx = Context({
            "episodes": 1,
            "render": False
        })

        avg_reward, avg_ep_length = algo.play(mock_env, ctx)
        mock_env.reset.assert_called_once()
        assert policy.act.call_count == 4
        assert mock_env.step.call_count == 4
        assert avg_reward == 10
        assert avg_ep_length == 4

    @mock.patch("gym.Env")
    def test_play_no_policy_should_fail(self, mock_env):
        algo = ReinforceContinuous()
        with pytest.raises(AssertionError) as e:
            algo.play(mock_env, 1)
            assert e.value == "The policy is not defined."

    @mock.patch("gym.Env")
    def test_train_no_policy_should_fail(self, mock_env):
        algo = ReinforceContinuous()
        ctx = Context({
            "episodes": MAX_TEST_EPISODES,
            "render": False
        })
        with pytest.raises(AssertionError) as e:
            algo.train(mock_env, ctx)
            assert e.value == "The policy is not defined."
