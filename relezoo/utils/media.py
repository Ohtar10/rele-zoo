import os
from typing import Union

import numpy as np
from moviepy.editor import ImageSequenceClip


def record_video(out_dir: str, frames: Union[list, np.ndarray], fps: int = 16,
                 file_name: str = "logging-video.mp4") -> str:
    clip = ImageSequenceClip(frames, fps=fps)
    file = os.path.join(out_dir, file_name)
    clip.write_videofile(file)
    return file
