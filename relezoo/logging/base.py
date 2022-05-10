import abc
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np


class Logging(abc.ABC):
    """Logging base class.
    """
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def log_scalar(self, name: str, data: Any, step: Optional[int] = None):
        pass

    @abstractmethod
    def log_image(self, name: str, data: Any, step: Optional[int] = None):
        pass

    @abstractmethod
    def log_video(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
        pass

    @abstractmethod
    def log_histogram(self, name: str, data: Any, step: Optional[int] = None):
        pass

    @abstractmethod
    def log_grads(self, model: Any, step: Optional[int] = None):
        pass

    @abstractmethod
    def log_video_from_frames(self,
                              name: str,
                              frames: Union[list, np.ndarray],
                              fps: int = 16,
                              step: Optional[int] = None):
        pass

