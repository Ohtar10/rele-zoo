import os
from typing import Any

from gym import Env
from hydra.utils import instantiate
from omegaconf import DictConfig

from relezoo.environments import GymWrapper


class Runner:
    """Runner.

    This class wraps all the run logic
    for experiments. It expects the
    necessary configurations from Hydra,
    then instantiates everything it needs,
    and finally runs the pipeline.
    """
    def __init__(self):
        self.workdir = os.getcwd()
        self.cfg = None
        self.environment = None
        self.algorithm = None
        self.logger = None

    def init(self, cfg: DictConfig) -> Any:
        """init.

        Initializes the experiment objects as
        per hydra configuration.
        """
        self.cfg = cfg
        self.environment: GymWrapper = instantiate(cfg.environment)
        self.logger = instantiate(cfg.logger,
                                  logdir=os.path.join(self.workdir, cfg.logger.logdir)
                                  )
        self.environment: GymWrapper = instantiate(cfg.environment)

        if self.cfg.network.infer_in_shape:
            # TODO this is not scalable. Expects specifics from the config
            in_shape = self.environment.get_observation_space()[0]
            self.cfg.algorithm.policy.network.in_shape = in_shape

        if self.cfg.network.infer_out_shape:
            # TODO this is not scalable. Expects specifics from the config
            out_shape = self.environment.get_action_space().n
            self.cfg.algorithm.policy.network.out_shape = out_shape

        env: Env = self.environment.build_env()
        self.algorithm = instantiate(self.cfg.algorithm, env=env, logger=self.logger)

    def run(self):
        """run.
        Runs the experiment as per
        configuration mode.
        """
        if "train" == self.cfg.mode:
            self.algorithm.train(self.cfg.episodes)
        elif "play" == self.cfg.mode:
            self.algorithm.play(self.cfg.episodes)


