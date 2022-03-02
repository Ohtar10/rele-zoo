import abc
from abc import abstractmethod
from typing import Any
import torch.nn as nn


class Network(abc.ABC, nn.Module):

    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass

    @abstractmethod
    def get_input_shape(self) -> Any:
        pass

    @abstractmethod
    def get_output_shape(self) -> Any:
        pass
