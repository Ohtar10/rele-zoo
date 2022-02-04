import os
import importlib
from typing import Any

from gym import Env
from hydra.utils import instantiate
from omegaconf import DictConfig

from relezoo.environments import GymWrapper


class Runner:
    def __init__(self):
        self.workdir = os.getcwd()
        self.cfg = None
        self.environment = None
        self.algorithm = None
        self.logger = None

    def init(self, cfg: DictConfig) -> Any:
        """
        # TODO
        1. Ensure work directory
        2. load all major elements based on class and properties
        2.1 Might suggest element loader for each? or a meta-loader that adapts
        3. Once all elements are loaded, submit the sequential call
        4. Collect and finish
        :param cfg:
        :return:
        """
        self.cfg = cfg
        self.environment: GymWrapper = instantiate(cfg.environment)
        self.logger = instantiate(cfg.logger,
                                  logdir=os.path.join(self.workdir, cfg.logger.logdir)
                                  )
        self.environment: GymWrapper = instantiate(cfg.environment)

        if self.cfg.network.infer_in_shape:
            in_shape = self.environment.get_observation_space()[0]
            self.cfg.algorithm.policy.network.in_shape = in_shape

        if self.cfg.network.infer_out_shape:
            out_shape = self.environment.get_action_space().n
            self.cfg.algorithm.policy.network.out_shape = out_shape

        env: Env = self.environment.build_env()
        self.algorithm = instantiate(self.cfg.algorithm, env=env, logger=self.logger)

    def run(self):
        if "train" == self.cfg.mode:
            self.algorithm.train(self.cfg.episodes)
        elif "play" == self.cfg.mode:
            self.algorithm.play(self.cfg.episodes)


