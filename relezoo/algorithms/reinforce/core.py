from typing import Optional

import numpy as np
import torch
from kink import inject

from relezoo.algorithms.base import Algorithm
from relezoo.algorithms.reinforce.discrete import ReinforceDiscretePolicy
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context


@inject
class Reinforce(Algorithm):
    """Reinforce
    Container class for all the necessary logic
    to train and use vanilla policy gradient aka REINFORCE
    with gym environments."""

    def __init__(
            self,
            logger: Logging,
            context: Context,
            policy: Optional[ReinforceDiscretePolicy] = None,
            batch_size: int = 5000):
        super(Reinforce, self).__init__(context, logger, batch_size, policy)
        self.train_steps = 0

    def _train_epoch(self,
                     env: Environment,
                     batch_size: int = 5000) -> (float, float, int):
        batch_obs = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lens = []

        obs = env.reset()
        episode_rewards = []

        while True:
            batch_obs.append(obs.copy())

            action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, reward, done, _ = env.step(action)

            batch_actions.append(action)
            episode_rewards.append(reward)

            if np.any(done):
                # On episode end, we must build the batch to submit
                # to later make the policy learn.
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lens.append(episode_length)

                # Despite this being called "weights" this is
                # actually R(tau), i.e., the return of the whole
                # trajectory, which is replicated through all the
                # episode steps as this is how the gradient
                # is calculated. See the gradient formula.
                batch_weights += [episode_return] * episode_length

                obs, done, episode_rewards = env.reset(), False, []

                if len(batch_obs) > batch_size:
                    break

        batch_loss = self.policy.learn(
            torch.from_numpy(np.array(batch_obs)),
            torch.from_numpy(np.array(batch_actions)),
            torch.from_numpy(np.array(batch_weights))
        )

        self._log(batch_loss, batch_returns, batch_lens)

        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def _log(self, batch_loss, batch_returns, batch_lens):
        if self.logger is not None:
            self.logger.log_scalar('training/loss', batch_loss, self.train_steps)
            self.logger.log_scalar('training/return', np.mean(batch_returns), self.train_steps)
            self.logger.log_scalar('training/episode_length', np.mean(batch_lens), self.train_steps)
            self.logger.log_grads(self.policy.net, self.train_steps)


