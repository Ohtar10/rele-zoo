import copy
from typing import Any, Dict, Optional
import random
import numpy as np


def make_noise(name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if name == 'ou-noise':
        return OUNoise(**params)
    elif name == 'None':
        return None


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size: int, mu: float = 0., theta: float = 0.15, sigma: float = 0.1, seed: Optional[int] = None):
        """Initialize parameters and noise process."""
        if seed is not None:
            random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
