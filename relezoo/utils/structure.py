from typing import Union, Any

from omegaconf import DictConfig


class Context:
    def __init__(self, config: Union[dict, DictConfig]):
        self._config = config
        self.__dict__.update(config.items())

    def __getitem__(self, key) -> Any:
        return self._config[key]



