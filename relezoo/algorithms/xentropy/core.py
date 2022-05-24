from typing import Optional

import numpy as np
import torch
from kink import inject

from relezoo.algorithms.base import Algorithm, Policy
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context


@inject
class CrossEntropyMethod(Algorithm):
    """Cross Entropy Method

    Container class for all the necessary logic
    to train and use a Cross Entropy Method
    based Algorithm.

    """

    def __init__(
            self,
            logger: Logging,
            context: Context,
            policy: Optional[Policy] = None,
            batch_size: int = 16,
            elite_percentile: float = 70.
    ):
        """

        Parameters
        ----------
        logger : Logging
            Logging mechanism to use.
        context : Context
            General experiment context parameters.
        policy : Policy
            The policy to train.
        batch_size : int
            The training batch size.
        elite_percentile : float
            The percentile threshold to select the elite rollouts.

        """
        super(CrossEntropyMethod, self).__init__(context, logger, batch_size, policy)
        self.elite_percentile = elite_percentile

    def train_epoch(self, env: Environment, batch_size: int) -> (float, float):
        batch = self.collect_trajectories(env, batch_size)
        train_obs, train_act, rewards, batch_lens = self._filter_elites(batch, self.elite_percentile)
        batch_loss = self.policy.learn(train_obs, train_act)

        self._log(batch_loss, rewards, batch_lens)
        self.train_steps += 1

        return batch_loss, rewards, batch_lens

    @staticmethod
    def _filter_elites(batch, percentile):
        rewards = [b.reward for b in batch]
        reward_bound = np.percentile(rewards, percentile)

        train_obs = []
        train_act = []
        elite_rewards = []
        elite_lens = []
        for episode in batch:
            if episode.reward < reward_bound:
                continue
            train_obs.extend([step.observation for step in episode.steps])
            train_act.extend([step.action for step in episode.steps])
            elite_rewards.append(episode.reward)
            elite_lens.append(len(episode.steps))

        train_obs = torch.tensor(np.array(train_obs))
        train_act = torch.tensor(np.array(train_act), dtype=torch.float)

        return train_obs, train_act, elite_rewards, elite_lens

    def _log(self, batch_loss, batch_returns, batch_lens):
        super(CrossEntropyMethod, self)._log(batch_loss, batch_returns, batch_lens)
        if self.logger is not None:
            self.logger.log_grads(self.policy.net, self.train_steps)
