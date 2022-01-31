from typing import Optional
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from relezoo.engine.runner import run


log = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='config')
def hrelezoo(cfg: Optional[DictConfig] = None) -> None:
    run(cfg)
    log.debug(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    hrelezoo()

