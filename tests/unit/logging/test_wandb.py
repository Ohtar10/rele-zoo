import mock
import numpy as np
import pytest
import torch
import torch.nn as nn

from relezoo.logging.wandb import WandbLogging
from kink import di

additional_config = {'additional': 'config'}


@pytest.fixture(autouse=True)
def run_around_tests():
    di['config'] = additional_config
    yield
    di.clear_cache()


@pytest.mark.unit
@mock.patch("relezoo.logging.wandb.wandb")
class TestWandbLogging:

    @pytest.mark.parametrize(
        "params",
        [
            {"project": "test"},
            {"project": "test", "name": "hello-world"}
        ]
    )
    def test_create_wandb_logging(self, mocked_wandb, params):
        logger = WandbLogging(**params)
        assert logger is not None
        assert mocked_wandb.init.call_count == 1
        full_params = {'config': additional_config}
        full_params.update(params)
        assert mocked_wandb.init.call_args.kwargs == full_params
        # should not fail

    def test_log_scalar(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(**params)
        logger.log_scalar("a_scalar", 0.5, 10)
        mocked_wandb.log.assert_called_once_with(
            {"a_scalar": 0.5, "step": 10}
        )

    def test_log_image(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(**params)
        tag = "an_image"
        data = np.random.rand(24, 24, 3)
        step = 10
        logger.log_image(tag, data, step)
        mocked_wandb.log.assert_called_once_with(
            {tag: mocked_wandb.Image(data), "step": step}
        )

    def test_log_video(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(**params)
        tag = "a_video"
        data = np.random.rand(10, 3, 24, 24)
        step = 10
        logger.log_video(tag, data, step)
        mocked_wandb.log.assert_called_once_with(
            {tag: mocked_wandb.Video(data), "step": step}
        )

    def test_log_histogram(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(**params)
        tag = "a_histogram"
        data = np.random.rand(3, 24, 24)
        step = 10
        logger.log_histogram(tag, data, step)
        mocked_wandb.log.assert_called_once_with(
            {tag: mocked_wandb.Histogram(data), "step": step}
        )

    def test_log_grads_watch(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(watch_grads=True, **params)
        mock_model = mock.MagicMock(nn.Module)
        mock_model.named_parameters.return_value = [
            ("dummy", mock.MagicMock(torch.Tensor))
        ]
        step = 10
        logger.log_grads(mock_model, step)
        mocked_wandb.watch.assert_called_once_with(mock_model)

    def test_log_grads_direct(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(**params)
        mock_model = mock.MagicMock(nn.Module)
        mock_model.named_parameters.return_value = [
            ("dummy", mock.MagicMock(torch.Tensor))
        ]
        step = 10
        logger.log_grads(mock_model, step)
        mocked_wandb.log.assert_called_once()

    def test_log_video_from_frames(self, mocked_wandb):
        params = {"project": "test"}
        logger = WandbLogging(**params)
        tag = "a_video"
        frames = np.random.randint(0, 255+1, (10, 24, 24, 3))
        frames = [f for f in frames]
        step = 10
        logger.log_video_from_frames(tag, frames, fps=16, step=step)
        mocked_wandb.log.assert_called_once_with(
            {tag: mocked_wandb.Video(), "step": step}
        )
