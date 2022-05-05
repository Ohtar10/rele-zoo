import logging
from typing import Optional, Any

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from relezoo.engine import Runner

log = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='config')
def hrelezoo(cfg: Optional[DictConfig] = None) -> Any:
    # print(OmegaConf.to_yaml(cfg))
    if not ray.is_initialized():
        ray.init(**cfg.ray.init)
    runner = Runner()
    runner.init(cfg)
    result = runner.run()
    log.debug(OmegaConf.to_yaml(cfg))
    ray.shutdown()
    return result


if __name__ == '__main__':
    hrelezoo()
