from collections import namedtuple
from typing import Optional, List, Any

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

    Episode = namedtuple('Episode', field_names=['reward', 'steps'])
    EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action', 'weight'])

    def __init__(
            self,
            logger: Logging,
            context: Context,
            policy: Optional[ReinforceDiscretePolicy] = None,
            batch_size: int = 5000):
        super(Reinforce, self).__init__(context, logger, batch_size, policy)
        self.train_steps = 0

    def _train_epoch(self, env: Environment, batch_size: int) -> (float, float, int):
        batch = []
        total_collected_steps = 0
        obs = env.reset()
        batch_obs = [False for _ in range(len(obs))]
        batch_actions = [False for _ in range(len(obs))]
        batch_rewards = [False for _ in range(len(obs))]

        while True:
            batch_obs = [
                np.concatenate([agent, ob.reshape(1, -1)]) if isinstance(agent, np.ndarray) else ob.reshape(1, -1)
                for agent, ob in zip(batch_obs, obs)
            ]
            actions = self.policy.act(torch.tensor(obs)).cpu().numpy()
            actions = actions.reshape(-1, 1)  # TODO discrete action of dim 1, and continuous?

            obs, rewards, dones, _ = env.step(actions)
            rewards = rewards.reshape(-1, 1)

            batch_actions = [
                np.concatenate([agent, action.reshape(1, -1)]) if isinstance(agent, np.ndarray)
                else action.reshape(1, -1)
                for agent, action in zip(batch_actions, actions)
            ]

            batch_rewards = [
                np.concatenate([agent, reward.reshape(1, -1)]) if isinstance(agent, np.ndarray)
                else reward.reshape(1, -1)
                for agent, reward in zip(batch_rewards, rewards)]

            if np.any(dones):
                # On episode end, we must build the batch to submit
                # to later make the policy learn.
                dones_idx = np.where(dones)[0]
                for idx in dones_idx:
                    # get the trajectory of this episode
                    episode_obs = batch_obs[idx]
                    episode_act = batch_actions[idx]
                    episode_return = np.sum(batch_rewards[idx])

                    # Despite this being called "weights" this is
                    # actually R(tau), i.e., the return of the whole
                    # trajectory, which is replicated through all the
                    # episode steps as this is how the gradient
                    # is calculated. See the gradient formula.
                    steps = [
                        Reinforce.EpisodeStep(episode_obs[i], episode_act[i], episode_return)
                        for i in range(len(episode_obs))
                    ]
                    total_collected_steps += len(steps)
                    batch.append(Reinforce.Episode(episode_return, steps))

                    # Reset the finished agent
                    obs[dones_idx, :] = env.reset(idx)
                    batch_obs[idx] = False
                    batch_actions[idx] = False
                    batch_rewards[idx] = False
                # Stop condition depends only on the number of steps retrieved
                # Not necessarily in the number of episodes
                if total_collected_steps >= batch_size:
                    break

        train_obs, train_act, train_weights, batch_returns, batch_lens = self._prepare_batch(batch)
        batch_loss = self.policy.learn(
            train_obs,
            train_act,
            train_weights
        )

        self._log(batch_loss, batch_returns, batch_lens)

        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    @staticmethod
    def _prepare_batch(batch: List[Episode]) -> Any:
        train_obs = []
        train_act = []
        train_weights = []
        batch_returns = []
        batch_lens = []
        for episode in batch:
            train_obs.extend([s.observation for s in episode.steps])
            train_act.extend([s.action for s in episode.steps])
            train_weights.extend([s.weight for s in episode.steps])
            batch_returns.append(episode.reward)
            batch_lens.append(len(episode.steps))

        train_obs = torch.tensor(np.array(train_obs))
        train_act = torch.tensor(np.array(train_act))
        train_weights = torch.tensor(np.array(train_weights))
        return train_obs, train_act, train_weights, batch_returns, batch_lens

    def _log(self, batch_loss, batch_returns, batch_lens):
        if self.logger is not None:
            self.logger.log_scalar('training/loss', batch_loss, self.train_steps)
            self.logger.log_scalar('training/return', np.mean(batch_returns), self.train_steps)
            self.logger.log_scalar('training/episode_length', np.mean(batch_lens), self.train_steps)
            self.logger.log_grads(self.policy.net, self.train_steps)


