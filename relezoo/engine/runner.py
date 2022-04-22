import os
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from relezoo.environments.base import Environment
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
        self.logger = None

    def init(self, cfg: DictConfig) -> Any:
        """init.

        Initializes the experiment objects as
        per hydra configuration.
        """
        self.cfg = cfg
        self.env_train: Environment = instantiate(cfg.env_train)
        self.env_test: Environment = instantiate(cfg.env_test)
        self.logger = instantiate(cfg.logger)

        if self.cfg.network.infer_in_shape:
            # TODO this is not scalable. Expects specifics from the config
            in_shape = self.env_train.get_observation_space()[1]
            self.cfg.algorithm.policy.network.in_shape = in_shape

        if self.cfg.network.infer_out_shape:
            # TODO this is not scalable. Expects specifics from the config
            out_shape = self.env_train.get_action_space()[1]
            self.cfg.algorithm.policy.network.out_shape = out_shape

        self.algorithm = instantiate(self.cfg.algorithm, logger=self.logger)

    def run(self) -> Any:
        """run.
        Runs the experiment as per
        configuration mode.
        """
        ctx = Context(self.cfg.context)
        result = None
        if "train" == ctx.mode:
            os.makedirs(Path(ctx.checkpoints), exist_ok=True)
            result = self.algorithm.train(self.env_train, ctx, self.env_test)
            self.algorithm.save(os.path.join(self.workdir, ctx.checkpoints))
        elif "play" == ctx.mode:
            self.algorithm.load(ctx.checkpoints, ctx)
            result = self.algorithm.play(self.env_test, ctx)

        return result


