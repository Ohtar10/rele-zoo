import tempfile
from typing import Union
import numpy as np
import os
from moviepy.editor import ImageSequenceClip


def record_video(frames: Union[list, np.ndarray], fps: int = 16) -> str:
    with tempfile.TemporaryDirectory() as td:
        clip = ImageSequenceClip(frames, fps=fps)
        file = os.path.join(td, 'logging-video.mp4')
        clip.write_videofile(file)
        return file
