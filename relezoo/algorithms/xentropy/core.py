from collections import namedtuple, deque
from typing import Optional

import numpy as np
import torch
from kink import inject

from relezoo.algorithms.base import Algorithm
from relezoo.algorithms.xentropy.discrete import CrossEntropyDiscretePolicy
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.utils.structure import Context


@inject
class CrossEntropyMethod(Algorithm):
    Episode = namedtuple('Episode', field_names=['reward', 'steps'])
    EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

    def __init__(
            self,
            logger: Logging,
            context: Context,
            policy: Optional[CrossEntropyDiscretePolicy] = None,
            batch_size: int = 16,
            elite_percentile: float = 70.
    ):
        super(CrossEntropyMethod, self).__init__(context, logger, batch_size, policy)
        self.elite_percentile = elite_percentile
        self.mean_reward_window = context.mean_reward_window
        self.train_steps = 0
        self.avg_return_pool = deque(maxlen=self.mean_reward_window)

    def _train_epoch(self,
                     env: Environment,
                     batch_size: int) -> (float, float):
        batch = []
        obs = env.reset()  # (#agents, obs_space)
        batch_obs = [False for _ in range(len(obs))]
        # initialize the rest of the batch elements as a falsy element
        batch_actions = [False for _ in range(len(obs))]
        batch_returns = [False for _ in range(len(obs))]

        while True:
            batch_obs = [
                np.concatenate([agent, ob.reshape(1, -1)]) if isinstance(agent, np.ndarray) else ob.reshape(1, -1)
                for agent, ob in zip(batch_obs, obs)
            ]
            actions = self.policy.act(torch.tensor(obs)).cpu().numpy()  # (#agents, action_space)
            actions = actions.reshape(-1, 1)  # discrete action of dim 1

            obs, rewards, dones, _ = env.step(actions)
            rewards = rewards.reshape(-1, 1)  # reward of dim 1

            batch_actions = [
                np.concatenate([agent, action.reshape(1, -1)]) if isinstance(agent, np.ndarray)
                else action.reshape(1, -1)
                for agent, action in zip(batch_actions, actions)
            ]

            batch_returns = [
                np.concatenate([agent, reward.reshape(1, -1)]) if isinstance(agent, np.ndarray)
                else reward.reshape(1, -1)
                for agent, reward in zip(batch_returns, rewards)]

            if np.any(dones):
                # get done idx
                dones_idx = np.where(dones)[0]
                for idx in dones_idx:
                    # Build an episode from this agent and add it to the batch
                    done_obs = batch_obs[idx]
                    done_act = batch_actions[idx]
                    done_return = np.sum(batch_returns[idx])
                    steps = [CrossEntropyMethod.EpisodeStep(done_obs[i], done_act[i]) for i in range(len(done_obs))]
                    batch.append(CrossEntropyMethod.Episode(done_return, steps))

                    # Reset the finished agent
                    obs[dones_idx, :] = env.reset(idx)
                    batch_obs[idx] = False
                    batch_actions[idx] = False
                    batch_returns[idx] = False

                if len(batch) >= batch_size:
                    break

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
        for reward, steps in batch:
            if reward < reward_bound:
                continue
            train_obs.extend([step.observation for step in steps])
            train_act.extend([step.action for step in steps])
            elite_rewards.append(reward)
            elite_lens.append(len(steps))

        train_obs = torch.tensor(np.array(train_obs))
        train_act = torch.tensor(np.array(train_act), dtype=torch.long)

        return train_obs, train_act, elite_rewards, elite_lens

    def _log(self, batch_loss, batch_returns, batch_lens):
        if self.logger is not None:
            self.logger.log_scalar('training/loss', batch_loss, self.train_steps)
            self.logger.log_scalar('training/return', np.mean(batch_returns), self.train_steps)
            self.logger.log_scalar('training/mean_episode_length', np.mean(batch_lens), self.train_steps)
            self.logger.log_grads(self.policy.net, self.train_steps)

    def load(self, save_path: str, context: Optional[Context] = None) -> None:
        device = "cuda" if context and context.gpu and torch.cuda.is_available() else "cpu"
        net = torch.load(save_path, map_location=torch.device(device))
        self.policy = CrossEntropyDiscretePolicy(net)
