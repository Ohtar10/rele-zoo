import abc
from abc import abstractmethod
from typing import Any, Optional, Union, List, Dict

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
    def log_scalar(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
        """Log a scalar value

        Parameters
        ----------
        name : str
            Name or tag of the metric or value.
        data : Any
            Actual value of the metric.
        step : Optional[int]
            Step number in which this metric was obtained.
        kwargs : dict
            Additional parameters. Usage subject to logging backend.

        """
        pass

    @abstractmethod
    def log_image(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
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
        kwargs : dict
            Additional parameters. Usage subject to logging backend.

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
    def log_histogram(self, name: str, data: Any, step: Optional[int] = None, **kwargs):
        """Log histogram

        Parameters
        ----------
        name : str
            Name or tag of the histogram.
        data : Any
            Histogram data.
        step : int
            Step number in which this metric was obtained.
        kwargs : dict
            Additional parameters. Usage subject to logging backend.

        """
        pass

    @abstractmethod
    def log_grads(self, model: Any, step: Optional[int] = None, **kwargs):
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
        kwargs : dict
            Additional parameters. Usage subject to logging backend.

        """
        pass

    @abstractmethod
    def log_video_from_frames(self,
                              name: str,
                              frames: Union[list, np.ndarray],
                              fps: int = 16,
                              step: Optional[int] = None,
                              auto_clean: bool = False):
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
        auto_clean : bool
            Determines if the generated video file should
            be automatically deleted after this call.

        """
        pass

    @abstractmethod
    def log_table_row(self, name: str, col_params: Dict[str, str], data: List[Any]):
        """Logs the provided data as a table.

        This method will inspect the provided data
        and depending on its data type and the provided
        column parameters, it will apply a transformation
        before generating a record.

        **Note:** This function will store a local
        copy of the table name provided, and it will
        add the provided data as a single record.
        This function WILL NOT submit the table!!
        It is expected any pending data to be submitted
        when invoking the flush operation in the logging.


        Parameters
        ----------
        name : str
            Table name key.
        data : List[Any]
            Raw data to be added as row. This must be a single ROW in the table
        col_params : Dict[str, List[Any]]
            Represent the columns and a transformation function per column to
            call to each data in the column.

        """
        pass

