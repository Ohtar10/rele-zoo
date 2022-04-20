from typing import Optional
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from relezoo.engine import Runner
import ray

log = logging.getLogger(__name__)


@hydra.main(config_path='../conf', config_name='config')
def hrelezoo(cfg: Optional[DictConfig] = None) -> None:
    # print(OmegaConf.to_yaml(cfg))
    ray.init(
        num_cpus=cfg.ray.cpus,
        object_store_memory=cfg.ray.memory,
        dashboard_port=cfg.ray.dashboard_port,
        ignore_reinit_error=True
    )
    runner = Runner()
    runner.init(cfg)
    runner.run()
    log.debug(OmegaConf.to_yaml(cfg))
    ray.shutdown()


if __name__ == '__main__':
    hrelezoo()

