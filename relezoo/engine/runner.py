import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
from hydra.utils import instantiate
from kink import di, inject
from omegaconf import DictConfig, OmegaConf

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

    def set_seed(self, seed: Optional[int]):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.env_train.seed(seed)
            self.env_test.seed(seed)

    def init(self, cfg: DictConfig) -> Any:
        """init.

        Initializes the experiment objects as
        per hydra configuration.
        """
        if 'num_envs' in cfg.env_train and not ray.is_initialized():
            ray.init(**cfg.ray.init)

        self.cfg = cfg
        self.env_train: Environment = instantiate(cfg.env_train)
        self.env_test: Environment = instantiate(cfg.env_test)
        self.set_seed(self.cfg.context.seed)

        di['config'] = OmegaConf.to_container(cfg)
        di[Logging] = instantiate(cfg.logger)
        di[Context] = Context(self.cfg.context)

        # TODO this is not scalable. Expects specifics from the config
        if self.cfg.network.infer_in_shape:
            in_shape = self.env_train.get_observation_space()[1]
            self.cfg.algorithm.policy.network.in_shape = in_shape

        if self.cfg.network.infer_out_shape:
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
        if "train" == ctx.mode or "resume" == ctx.mode:
            self.log.info("Running Training Session with config:")
            self.log.info(f"\n{OmegaConf.to_yaml(self.cfg)}")
            self.log.info("Press CTRL + C to cancel the run")
            self.log.info("A checkpoint will be saved automatically after a successful run or cancel.")
            if "resume" == ctx.mode:
                self.algorithm.load(ctx.resume_from)

            os.makedirs(Path(ctx.checkpoints), exist_ok=True)
            try:
                msg, result = self.algorithm.train(self.env_train, self.env_test)
                self.log.info(msg)
            except KeyboardInterrupt:
                self.log.info("Training interrupted.")
            finally:
                self.log.info("Saving current progress...")
                self.algorithm.save(os.path.join(self.workdir, ctx.checkpoints))
        elif "play" == ctx.mode:
            # Baselines are expected to be three directories up
            if ctx.checkpoints.startswith("baselines"):
                ctx.checkpoints = f"../../../{ctx.checkpoints}"

            self.algorithm.load(ctx.checkpoints)
            try:
                result, _, _ = self.algorithm.play(self.env_test)
            except KeyboardInterrupt:
                self.log.info("Play interrupted...")

        di.clear_cache()
        return result

    @staticmethod
    def teardown():
        if ray.is_initialized():
            ray.shutdown()

