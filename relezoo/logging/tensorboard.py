import importlib
import os.path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter as TXSummaryWriter
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

from relezoo.logging.base import Logging


class TensorboardLogging(Logging):

    def __init__(self, workdir: str, backend_class: Union[str, TXSummaryWriter, TorchSummaryWriter], log_dir: str):
        super(TensorboardLogging, self).__init__()
        if isinstance(backend_class, str):
            backend_module = ".".join(backend_class.split(".")[:-1])
            backend_class = backend_class.split(".")[-1]
            backend_module = importlib.import_module(backend_module)
            clazz = getattr(backend_module, backend_class)
            self.backend: Union[TXSummaryWriter, TorchSummaryWriter] = \
                clazz(os.path.join(workdir, log_dir))
        else:
            self.backend = backend_class

    def init(self):
        # Don't need to do anything with Tensorboard logger
        pass

    def flush(self):
        self.backend.flush()

    def close(self):
        self.backend.close()

    def log_scalar(self, name: str, data: Any, step: Optional[int] = None):
        self.backend.add_scalar(
            tag=name,
            scalar_value=data,
            global_step=step
        )

    def log_image(self, name: str, data: Any, step: Optional[int] = None):
        self.backend.add_image(
            tag=name,
            img_tensor=data,
            global_step=step
        )

    def log_video(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
        self.backend.add_video(
            tag=name,
            vid_tensor=data,
            global_step=step,
            **kwargs
        )

    def log_histogram(self, name: str, data: Any, step: Optional[int] = None):
        self.backend.add_histogram(
            tag=name,
            values=data,
            global_step=step
        )

    def log_grads(self, model: Any, step: Optional[int] = None):
        assert isinstance(model, nn.Module), "model must be a torch module"
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            name = name.replace(".", "/")
            self.log_histogram(
                f"grads/{name}",
                p.grad.detach().cpu().numpy(),
                step
            )

    def log_video_from_frames(self,
                              name: str,
                              frames: Union[list, np.ndarray],
                              fps: int = 16,
                              step: Optional[int] = None):
        # T x H x W x C
        sequence = np.array(frames) if isinstance(frames, list) else frames
        # T x C x H x W
        sequence = np.transpose(sequence, [0, 3, 1, 2])
        # B x T x C x H x W
        sequence = np.expand_dims(sequence, axis=0)
        sequence = torch.tensor(sequence)
        self.log_video(
            name,
            sequence,
            step,
            fps=fps
        )






