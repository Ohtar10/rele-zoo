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
        self.train_steps = max(0, context.start_at_step - 1)
        self.use_reward_2go = reward_2go

    def train_epoch(self, env: Environment, batch_size: int) -> (float, float, int):
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
                    rewards_2go = []
                    if self.use_reward_2go:
                        rewards_2go = self._reward_2go(batch_rewards[idx])

                    # Despite this being called "weights" this is
                    # actually R(tau), i.e., the return of the whole
                    # trajectory, which is replicated through all the
                    # episode steps as this is how the gradient
                    # is calculated. See the gradient formula.
                    steps = [
                        EpisodeStep(
                            episode_obs[i],
                            episode_act[i],
                            rewards_2go[i] if self.use_reward_2go else episode_return
                        )
                        for i in range(len(episode_obs))
                    ]
                    total_collected_steps += len(steps)
                    batch.append(Episode(episode_return, steps))

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
        return rtgs


