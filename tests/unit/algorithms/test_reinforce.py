import mock
import numpy as np

from relezoo.algorithms.reinforce import Reinforce


@mock.patch("tensorboardX.SummaryWriter")
@mock.patch("gym.Env")
@mock.patch("relezoo.algorithms.reinforce.Policy")
def test_reinforce_train(mock_policy, mock_env, mock_logger):
    algo = Reinforce(mock_env, mock_policy, mock_logger)
    algo._train_epoch = mock.MagicMock(return_value=(0.1, np.array([1, 2]), np.array([1, 2])))
    algo.train()
    assert algo._train_epoch.call_count == algo.epochs
    assert mock_logger.flush.call_count == algo.epochs
    mock_logger.close.assert_called_once()


