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
        num_cpus=cfg.ray_cpus,
        object_store_memory=cfg.ray_memory,
        dashboard_port=cfg.ray_dashboard_port
    )
    runner = Runner()
    runner.init(cfg)
    runner.run()
    log.debug(OmegaConf.to_yaml(cfg))
    ray.shutdown()


if __name__ == '__main__':
    hrelezoo()

