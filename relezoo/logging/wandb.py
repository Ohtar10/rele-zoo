import os.path
import re
import tempfile
from typing import Union, Optional, Any, Dict, List

import numpy as np
import torch.nn as nn
import wandb
from kink import inject

from relezoo.logging.base import Logging
from relezoo.utils.media import record_video


@inject
class WandbLogging(Logging):
    """Wandb implementation for :py:class:`relezoo.logging.base.Logging`

    This class implements the base Logging API using wandb as logging
    mechanism. All special interactions, e.g., init, wandb types
    transformation is handled here.


    """

    def __init__(self, config, watch_grads: bool = False, **kwargs):
        self.config = config
        self.params = kwargs
        self.watch_grads = watch_grads
        self.watching_grads = False
        self.__tables = {}
        self.__local_folder = tempfile.TemporaryDirectory()
        self.init()

    def init(self):
        wandb.init(config=self.config, **self.params)

    def flush(self):
        pass

    def close(self):
        for k, v in self.__tables.items():
            wandb.log({k: v})
        self.__local_folder.cleanup()
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
                              step: Optional[int] = None,
                              auto_clean: bool = False):
        frames = frames if isinstance(frames, list) else [f for f in frames]
        if auto_clean:
            with tempfile.TemporaryDirectory() as td:
                video_file = record_video(td, frames, fps)
                self.log_video(name=name, data=video_file, step=step)
        else:
            video_file = record_video(
                self.__local_folder.name,
                frames,
                fps=fps,
                file_name=f"logging-video-{step}.mp4"
            )
            self.log_video(name=name, data=video_file, step=step)

    def log_table_row(self, name: str, col_params: Dict[str, str], data: List[Any]):
        if name in self.__tables:
            table = self.__tables[name]
        else:
            table = wandb.Table(columns=list(col_params.keys()))
            self.__tables[name] = table

        assert len(col_params) == len(data)
        row = []
        for op, value in zip(col_params.values(), data):
            row.append(self.__transform(op, value))

        table.add_data(*row)

    def __transform(self, operation: str, value: Any) -> Any:
        if operation == 'noop':
            return value
        elif re.match(r'video\((.+)\)', operation) is not None:
            args = re.match(r'video\((.+)\)', operation).group(1)
            args = args.split(',')  # frames, file_name
            video_file = record_video(self.__local_folder.name, value, fps=int(args[0]), file_name=args[1])
            return wandb.Video(video_file)
        elif re.match(r'video_file\((.+)\)', operation) is not None:
            file_name = re.match(r'video_file\((.+)\)', operation).group(1)
            return wandb.Video(os.path.join(self.__local_folder.name, file_name))



