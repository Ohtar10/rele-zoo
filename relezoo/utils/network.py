from enum import Enum


class NetworkMode(Enum):
    """Network Mode enumerator.

    Represents two modes of network:
    # TRAIN: The network is in training mode, hence learning.
    # EVAL: The network is in evaluation mode.
    """
    TRAIN = 1
    EVAL = 2
