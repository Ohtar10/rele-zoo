import mock
import numpy as np
import pytest
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter as TXSummaryWriter
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
from relezoo.logging.tensorboard import TensorboardLogging


@pytest.mark.unit
@pytest.mark.skip("tensorboard failing")
class TestTensorboardLogging:

    @pytest.mark.parametrize(
        "backend_class",
        [
            "torch.utils.tensorboard.writer.SummaryWriter",
            "tensorboardX.SummaryWriter"
        ]
    )
    def test_create_tensorboard_logging(self, backend_class):
        logger = TensorboardLogging(backend_class, "tensorboard")
        assert logger is not None
        assert logger.backend is not None
        assert isinstance(logger.backend, TXSummaryWriter) or isinstance(logger.backend, TorchSummaryWriter)

    @mock.patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_scalar(self, mocked_sw):
        logger = TensorboardLogging(mocked_sw, "tb")
        logger.log_scalar("a_scalar", 0.5, 10)
        mocked_sw.add_scalar.assert_called_once_with(
            tag="a_scalar",
            scalar_value=0.5,
            global_step=10)

    @mock.patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_image(self, mocked_sw):
        logger = TensorboardLogging(mocked_sw, "tb")
        tag = "an_image"
        data = np.random.rand(3, 24, 24)
        step = 10
        logger.log_image(tag, data, step)
        mocked_sw.add_image.assert_called_once_with(
            tag=tag,
            img_tensor=data,
            global_step=step)

    @mock.patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_video(self, mocked_sw):
        logger = TensorboardLogging(mocked_sw, "tb")
        tag = "a_video"
        data = np.random.rand(10, 3, 24, 24)
        step = 10
        logger.log_video(tag, data, step)
        mocked_sw.add_video.assert_called_once_with(
            tag=tag,
            vid_tensor=data,
            global_step=step)

    @mock.patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_histogram(self, mocked_sw):
        logger = TensorboardLogging(mocked_sw, "tb")
        tag = "a_histogram"
        data = np.random.rand(3, 24, 24)
        step = 10
        logger.log_histogram(tag, data, step)
        mocked_sw.add_histogram.assert_called_once_with(
            tag=tag,
            values=data,
            global_step=step)

    @mock.patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_grads(self, mocked_sw):
        logger = TensorboardLogging(mocked_sw, "tb")
        mock_model = mock.MagicMock(nn.Module)
        mock_model.named_parameters.return_value = [
            ("dummy", mock.MagicMock(torch.Tensor))
        ]
        step = 10

        logger.log_grads(mock_model, step)
        mocked_sw.add_histogram.assert_called_once()

    @mock.patch("torch.utils.tensorboard.SummaryWriter")
    def test_log_video_from_frames(self, mocked_sw):
        logger = TensorboardLogging(mocked_sw, "tb")
        tag = "a_video"
        frames = [np.random.rand(24, 24, 3) for _ in range(10)]
        step = 10
        logger.log_video_from_frames(tag, frames, step)
        mocked_sw.add_video.assert_called_once()

