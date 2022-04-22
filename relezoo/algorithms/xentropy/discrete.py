import os
from collections import namedtuple, deque
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from relezoo.algorithms.base import Algorithm, Policy
from relezoo.environments.base import Environment
from relezoo.logging.base import Logging
from relezoo.networks.base import Network
from relezoo.utils.network import NetworkMode
from relezoo.utils.structure import Context


class CrossEntropyDiscretePolicy(Policy):

    def __init__(self,
                 network: Network,
                 learning_rate: float = 1e-2):
        self.net = network
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)
        self.device = 'cpu'

    def set_mode(self, mode: NetworkMode):
        if mode == NetworkMode.TRAIN:
            self.net.train()
        else:
            self.net.eval()

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        obs = obs.to(self.device)
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits).sample()

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: Optional[torch.Tensor] = None):
        batch_obs = batch_obs.to(self.device)
        batch_actions = torch.squeeze(batch_actions).to(self.device)
        self.optimizer.zero_grad()
        pred_actions = self.net(batch_obs)
        loss = self.objective(pred_actions, batch_actions)
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, save_path):
        path = os.path.join(save_path, f"{self.__class__.__name__}.cpt")
        torch.save(self.net, path)

    def to(self, device: str) -> None:
        self.device = device
        self.net = self.net.to(device)


class CrossEntropyDiscrete(Algorithm):
    Episode = namedtuple('Episode', field_names=['reward', 'steps'])
    EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

    def __init__(
            self,
            policy: Optional[CrossEntropyDiscretePolicy] = None,
            batch_size: int = 16,
            elite_percentile: float = 70.,
            mean_return_window: int = 100,
            logger: Optional[Logging] = None
    ):
        self.policy = policy
        self.batch_size = batch_size
        self.logger = logger
        self.elite_percentile = elite_percentile
        self.mean_return_window = mean_return_window
        self.train_steps = 0
        self.avg_return_pool = deque(maxlen=mean_return_window)

    def train(self, env: Environment, context: Context, eval_env: Optional[Environment] = None) -> Any:
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.TRAIN)
        epochs = context.epochs
        render = context.render
        device = "cuda" if context.gpu and torch.cuda.is_available() else "cpu"
        self.policy.to(device)
        with tqdm(total=epochs) as progress:
            for i in range(1, epochs + 1):
                is_last_epoch = i == epochs
                batch_loss, batch_returns, batch_lens = self._train_epoch(env, self.batch_size)
                if eval_env is not None and (i % context.eval_every == 0 or is_last_epoch):  # evaluate every 10 epochs
                    self._evaluate(eval_env, render)
                progress.set_postfix({
                    "loss": f"{batch_loss:.2f}",
                    "score": f"{np.mean(batch_returns):.2f}",
                    "mean_length": f"{np.mean(batch_lens):.2f}"
                })
                progress.update()
                if self.logger is not None:
                    self.logger.flush()
        if self.logger is not None:
            self.logger.close()

        return f"Mean Return over {self.mean_return_window} epochs: {np.mean(self.avg_return_pool):.2f}"

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
                    steps = [CrossEntropyDiscrete.EpisodeStep(done_obs[i], done_act[i]) for i in range(len(done_obs))]
                    batch.append(CrossEntropyDiscrete.Episode(done_return, steps))

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

    def _evaluate(self, env: Environment, render: bool = False):
        render_frames = []
        episode_return = 0
        obs = env.reset()
        while True:
            if render:
                render_frames.append(env.render(mode='rgb_array'))

            action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                break

        self.avg_return_pool.append(episode_return)
        self.logger.log_scalar("evaluation/return", episode_return, step=self.train_steps)
        self.logger.log_scalar('evaluation/episode_length', len(render_frames), step=self.train_steps)
        self.logger.log_scalar(f'evaluation/mean_return_over_{self.mean_return_window}_episodes',
                               np.mean(self.avg_return_pool), step=self.train_steps)

        if render:
            self.logger.log_video_from_frames("live-play", render_frames, fps=16, step=self.train_steps)

    def play(self, env: Environment, context: Context) -> (float, int):
        assert self.policy is not None, "The policy is not defined."
        self.policy.set_mode(NetworkMode.EVAL)
        episodes = context.epochs
        render = context.render
        device = "cuda" if context.gpu and torch.cuda.is_available() else "cpu"
        self.policy.to(device)
        with tqdm(total=episodes) as progress:
            ep_rewards = []
            ep_lengths = []
            for i in range(1, episodes + 1):
                obs = env.reset()
                ep_length = 1
                ep_reward = 0
                while True:
                    if render:
                        env.render()

                    action = self.policy.act(torch.from_numpy(obs)).cpu().numpy()
                    obs, reward, done, _ = env.step(action)
                    ep_reward += reward
                    if done:
                        break
                    ep_length += 1
                ep_lengths.append(ep_length)
                ep_rewards.append(ep_reward)
                progress.set_postfix({
                    "Avg. reward": f"{np.mean(ep_rewards):.2f}",
                    "Avg. ep length": f"{np.mean(ep_length):.2f}"
                })
                progress.update()
        return np.mean(ep_rewards), np.mean(ep_length)

    def _log(self, batch_loss, batch_returns, batch_lens):
        if self.logger is not None:
            self.logger.log_scalar('training/loss', batch_loss, self.train_steps)
            self.logger.log_scalar('training/return', np.mean(batch_returns), self.train_steps)
            self.logger.log_scalar('training/mean_episode_length', np.mean(batch_lens), self.train_steps)
            self.logger.log_grads(self.policy.net, self.train_steps)

    def save(self, save_path: str) -> None:
        self.policy.save(save_path)

    def load(self, save_path: str, context: Optional[Context] = None) -> None:
        device = "cuda" if context and context.gpu and torch.cuda.is_available() else "cpu"
        net = torch.load(save_path, map_location=torch.device(device))
        self.policy = CrossEntropyDiscretePolicy(net)
