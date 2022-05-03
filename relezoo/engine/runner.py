import os
from kink import di, inject
from pathlib import Path
from typing import Any
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context


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
        self.env_train = None
        self.env_test = None
        self.algorithm = None
        self.log = logging.getLogger(__name__)

    def init(self, cfg: DictConfig) -> Any:
        """init.

        Initializes the experiment objects as
        per hydra configuration.
        """
        self.cfg = cfg
        self.env_train: Environment = instantiate(cfg.env_train)
        self.env_test: Environment = instantiate(cfg.env_test)

        di['config'] = cfg
        di[Logging] = instantiate(cfg.logger)
        di[Context] = Context(self.cfg.context)

        if self.cfg.network.infer_in_shape:
            # TODO this is not scalable. Expects specifics from the config
            in_shape = self.env_train.get_observation_space()[1]
            self.cfg.algorithm.policy.network.in_shape = in_shape

        if self.cfg.network.infer_out_shape:
            # TODO this is not scalable. Expects specifics from the config
            out_shape = self.env_train.get_action_space()[1]
            self.cfg.algorithm.policy.network.out_shape = out_shape

        self.algorithm = instantiate(self.cfg.algorithm)

    @inject
    def run(self, ctx: Context) -> Any:
        """run.
        Runs the experiment as per
        configuration mode.
        """
        result = None
        if "train" == ctx.mode:
            os.makedirs(Path(ctx.checkpoints), exist_ok=True)
            try:
                result = self.algorithm.train(self.env_train, self.env_test)
            except KeyboardInterrupt:
                self.log.info("Training interrupted. Saving current progress...")
            finally:
                self.algorithm.save(os.path.join(self.workdir, ctx.checkpoints))
        elif "play" == ctx.mode:
            self.algorithm.load(ctx.checkpoints)
            try:
                result = self.algorithm.play(self.env_test)
            except KeyboardInterrupt:
                self.log.info("Play interrupted...")

        di.clear_cache()
        return result


