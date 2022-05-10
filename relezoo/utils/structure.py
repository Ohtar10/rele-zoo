from typing import Union, Any

from omegaconf import DictConfig


class Context:
    """Run Context

    This is a container object with
    general parameters for a run meant
    to be shared among different elements
    of the run.

    """
    def __init__(self, config: Union[dict, DictConfig]):
        """

        Parameters
        ----------
        config : Union[dict, DictConfig]
            Source of context parameters
        """
        self._config = config
        self.__dict__.update(config.items())

    def __getitem__(self, key) -> Any:
        return self._config[key]



