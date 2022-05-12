import abc
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np


class Logging(abc.ABC):
    """Logging base class.

    This class serves as a contract for
    the logging backends for an experiment run.
    In essence, every logging backend meant to
    be used should implement the methods in this
    class as deemed necessary. It is expected that
    if certain operation is not available in the
    logging backend or is not necessary,
    the implementation should be a simple No-Op.

    The operations here are inspired by the default
    tensorboard backend.

    """
    @abstractmethod
    def init(self):
        """Runs initialization routine.

        """
        pass

    @abstractmethod
    def flush(self):
        """Flushes any cached information into the underlying backend."""
        pass

    @abstractmethod
    def close(self):
        """Closes any open connections or files."""
        pass

    @abstractmethod
    def log_scalar(self, name: str, data: Any, step: Optional[int] = None):
        """Log a scalar value

        Parameters
        ----------
        name : str
            Name or tag of the metric or value.
        data : Any
            Actual value of the metric.
        step : Optional[int]
            Step number in which this metric was obtained.

        """
        pass

    @abstractmethod
    def log_image(self, name: str, data: Any, step: Optional[int] = None):
        """Log an image

        Parameters
        ----------
        name : str
            Name or tag of the image.
        data : Any
            Image data, usually a numpy array,
            but it could be different depending on the backend.
        step : Optional[int]
            Step number in which this metric was obtained.

        """
        pass

    @abstractmethod
    def log_video(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
        """Log a video

        Parameters
        ----------
        name : str
            Name or tag of the video.
        data : Any
            Video data, usually a tensor representing the video frames,
            but it could be different depending on the backend.
        step : Optional[int]
            Step number in which this metric was obtained.
        kwargs : dict
            Additional parameters for video processing.

        """
        pass

    @abstractmethod
    def log_histogram(self, name: str, data: Any, step: Optional[int] = None):
        """Log histogram

        Parameters
        ----------
        name : str
            Name or tag of the histogram.
        data : Any
            Histogram data.
        step : int
            Step number in which this metric was obtained.

        """
        pass

    @abstractmethod
    def log_grads(self, model: Any, step: Optional[int] = None):
        """Log gradients from a pytorch model.

        This method will automatically explore the provided
        model and log the gradients produced during its training
        at the moment of invocation.

        Parameters
        ----------
        model : Any
            The pytorch model.
        step : int
            Step number in which this metric was obtained.

        """
        pass

    @abstractmethod
    def log_video_from_frames(self,
                              name: str,
                              frames: Union[list, np.ndarray],
                              fps: int = 16,
                              step: Optional[int] = None):
        """Log video from frames

        This method will generate the video from the
        provided frames and automatically call :py:func:`self.log_video`
        with the corresponding values.

        Parameters
        ----------
        name : str
            Name or tag of the video.
        frames : Union[list, np.ndarray]
            List of numpy arrays or a full numpy array
            with the video frames.
        fps : int
            Frames per second to generate the video.
        step : int
            Step number in which this metric was obtained.


        """
        pass

