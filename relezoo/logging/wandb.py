import os.path
from typing import Union, Optional, Any
import tempfile
from moviepy.editor import ImageSequenceClip
import torch.nn as nn
import numpy as np
import wandb
from relezoo.logging.base import Logging


class WandbLogging(Logging):

    def __init__(self, watch_grads: bool = False, **kwargs):
        wandb.init(**kwargs)
        self.watch_grads = watch_grads
        self.watching_grads = False

    def init(self):
        pass

    def flush(self):
        pass

    def close(self):
        wandb.finish()

    def log_scalar(self, name: str, data: Any, step: Optional[int] = None):
        wandb.log({name: data, "step": step})

    def log_image(self, name: str, data: Any, step: Optional[int] = None):
        wandb.log({name: wandb.Image(data), "step": step})

    def log_video(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
        wandb.log({name: wandb.Video(data), "step": step})

    def log_histogram(self, name: str, data: Any, step: Optional[int] = None):
        wandb.log({name: wandb.Histogram(data), "step": step})

    def log_grads(self, model: Any, step: Optional[int] = None):
        assert isinstance(model, nn.Module), "model must be a torch module"
        if self.watch_grads and not self.watching_grads:
            wandb.watch(model)
            self.watching_grads = True
        else:
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                name = name.replace(".", "/")
                self.log_histogram(
                    f"grads/{name}",
                    p.grad.detach().cpu().numpy(),
                    step
                )

    def log_video_from_frames(self, name: str,
                              frames: Union[list, np.ndarray],
                              fps: int = 16,
                              step: Optional[int] = None):
        frames = frames if isinstance(frames, list) else [f for f in frames]
        with tempfile.TemporaryDirectory() as td:
            clip = ImageSequenceClip(frames, fps=fps)
            file = os.path.join(td, 'logging-video.mp4')
            clip.write_videofile(file)
            self.log_video(name=name, data=file, step=step)

