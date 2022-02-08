from typing import Optional
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from relezoo.engine import Runner


log = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='config')
def hrelezoo(cfg: Optional[DictConfig] = None) -> None:
    # print(OmegaConf.to_yaml(cfg))
    runner = Runner()
    runner.init(cfg)
    runner.run()
    log.debug(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    hrelezoo()

