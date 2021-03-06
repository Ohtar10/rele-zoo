from typing import Optional, List, Any

import numpy as np
import torch
from kink import inject

from relezoo.algorithms.base import Algorithm, Policy
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context, Episode, EpisodeStep


@inject
class Reinforce(Algorithm):
    """Reinforce

    Container class for all the necessary logic
    to train and use vanilla policy gradient aka REINFORCE
    with gym environments.

    """
    def __init__(
            self,
            logger: Logging,
            context: Context,
            policy: Optional[Policy] = None,
            batch_size: int = 500,
            reward_2go: bool = False):
        """

        Parameters
        ----------
        logger : Logging
            Logging mechanism to use.
        context : Context
            General experiment context parameters.
        policy : Policy
            The policy to train.
        batch_size: int
            The training batch size.
        reward_2go : bool
            Enable or disable reward to go for episodes.
        """
        super(Reinforce, self).__init__(context, logger, batch_size, policy)
        self.use_reward_2go = reward_2go

    def train_epoch(self, env: Environment, batch_size: int) -> (float, float, int):
        batch = self.collect_trajectories(env, batch_size)
        train_obs, train_act, train_weights, batch_returns, batch_lens = self._prepare_batch(batch)
        batch_loss = self.policy.learn(
            train_obs,
            train_act,
            train_weights
        )

        self._log(batch_loss, batch_returns, batch_lens)

        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def _prepare_batch(self, batch: List[Episode]) -> Any:
        train_obs = []
        train_act = []
        train_weights = []
        batch_returns = []
        batch_lens = []
        for episode in batch:
            train_obs.extend([s.observation for s in episode.steps])
            train_act.extend([s.action for s in episode.steps])

            if self.use_reward_2go:
                rewards_2go = self._reward_2go([s.reward for s in episode.steps])
                train_weights.extend(rewards_2go)
            else:
                train_weights.extend([episode.reward for _ in range(len(episode.steps))])

            batch_returns.append(episode.reward)
            batch_lens.append(len(episode.steps))

        train_obs = torch.tensor(np.array(train_obs))
        train_act = torch.tensor(np.array(train_act)).squeeze()
        train_weights = torch.tensor(np.array(train_weights))
        return train_obs, train_act, train_weights, batch_returns, batch_lens

    def _log(self, batch_loss, batch_returns, batch_lens):
        super(Reinforce, self)._log(batch_loss, batch_returns, batch_lens)
        if self.logger is not None:
            self.logger.log_grads(self.policy.net, self.train_steps)

    @staticmethod
    def _reward_2go(rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i + 1] if i+1 < n else 0)
        return [r[0] for r in rtgs]


