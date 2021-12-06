from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from gym import Env


class Policy:
    def __init__(self, net: nn.Module, learning_rate: float):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        logits = self.__get_policy(obs)
        action = logits.sample().item()
        return action

    def __get_policy(self, obs: torch.Tensor):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: torch.Tensor):
        self.optimizer.zero_grad()
        batch_loss = self.__compute_loss(batch_obs, batch_actions, batch_weights)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss

    def __compute_loss(self, obs, actions, weights):
        logp = self.__get_policy(obs).log_prob(actions)
        return -(logp * weights).mean()


class Reinforce:

    def __init__(self, env: Env, net: nn.Module, logger: Optional[SummaryWriter] = None, epochs: int = 50,
                 batch_size: int = 5000,
                 learning_date: float = 1e-2,
                 render: bool = False):
        self.env = env
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.n
        self.policy = Policy(net, learning_date)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_date
        self.render = render
        self.logger = logger
        self.train_steps = 0

    def train(self):
        with tqdm(total=self.epochs) as progress:
            for i in range(1, self.epochs + 1):
                batch_loss, batch_returns, batch_lens = self.__train_epoch()
                progress.set_postfix({
                    "loss": f"{batch_loss:.2f}",
                    "score": f"{np.mean(batch_returns):.2f}",
                    "episode_length": f"{np.mean(batch_lens)}"
                })
                progress.update()

    def __train_epoch(self) -> (float, float, int):
        batch_obs = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lens = []

        obs = self.env.reset()
        episode_rewards = []
        render_epoch = False

        while True:
            if self.render and not render_epoch:
                self.env.render()

            batch_obs.append(obs.copy())

            action = self.policy.act(torch.from_numpy(obs))
            obs, reward, done, _ = self.env.step(action)

            batch_actions.append(action)
            episode_rewards.append(reward)

            if done:
                episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lens.append(episode_length)

                batch_weights += [episode_return] * episode_length

                obs, done, episode_rewards = self.env.reset(), False, []

                render_epoch = True

                if len(batch_obs) > self.batch_size:
                    break

        batch_loss = self.policy.learn(
            torch.from_numpy(batch_obs),
            torch.from_numpy(batch_actions),
            torch.from_numpy(batch_weights)
        )

        if self.logger is not None:
            self.logger.add_scalars(f'{self.env.spec.id}', {
                'loss': batch_loss,
                'return': np.mean(batch_returns),
                'episode_length': np.mean(batch_lens)
            }, global_step=self.train_steps)
        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def play(self):
        pass
