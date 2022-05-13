import abc
from abc import abstractmethod
from typing import Any
import torch.nn as nn


class Network(abc.ABC, nn.Module):
    """Neural Network base class.

    This class serves a contract to implement
    neural networks compatible with the
    policies as they expect them to have
    specific methods.

    """

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Perform a forward pass using input x."""
        pass

    @abstractmethod
    def get_input_shape(self) -> Any:
        """Get the input shape defined for this network."""
        pass

    @abstractmethod
    def get_output_shape(self) -> Any:
        """Get the output shape defined for this network."""
        pass
