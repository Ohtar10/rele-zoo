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

    def __init__(self, env: Env, policy: Policy, logger: Optional[SummaryWriter] = None, epochs: int = 50,
                 batch_size: int = 5000,
                 render: bool = False):
        self.env = env
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.n
        self.policy = policy
        self.epochs = epochs
        self.batch_size = batch_size
        self.render = render
        self.logger = logger
        self.train_steps = 0

    def train(self):
        with tqdm(total=self.epochs) as progress:
            for i in range(1, self.epochs + 1):
                batch_loss, batch_returns, batch_lens = self._train_epoch()
                progress.set_postfix({
                    "loss": f"{batch_loss:.2f}",
                    "score": f"{np.mean(batch_returns):.2f}",
                    "episode_length": f"{np.mean(batch_lens):.2f}"
                })
                progress.update()
                if self.logger is not None:
                    self.logger.flush()
        if self.logger is not None:
            self.logger.close()

    def _train_epoch(self) -> (float, float, int):
        batch_obs = []
        batch_actions = []
        batch_weights = []
        batch_returns = []
        batch_lens = []

        obs = self.env.reset()
        episode_rewards = []
        render_epoch = False
        render_frames = []

        while True:
            if self.render and not render_epoch:
                self.env.render()

            if not render_epoch:
                render_frames.append(self.env.render(mode='rgb_array'))

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
            torch.from_numpy(np.array(batch_obs)),
            torch.from_numpy(np.array(batch_actions)),
            torch.from_numpy(np.array(batch_weights))
        )

        self.__log(batch_loss, batch_returns, batch_lens, render_frames)

        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def __log(self, batch_loss, batch_returns, batch_lens, render_frames):
        if self.logger is not None:
            self.logger.add_scalar('loss', batch_loss, self.train_steps)
            self.logger.add_scalar('return', np.mean(batch_returns), self.train_steps)
            self.logger.add_scalar('episode_length', np.mean(batch_lens), self.train_steps)

            for name, p in self.policy.net.named_parameters():
                if p.grad is None:
                    continue
                name = name.replace(".", "/")
                self.logger.add_histogram(
                    tag=f"grads/{name}", values=p.grad.detach().cpu().numpy(), global_step=self.train_steps
                )

            # Render video every 10 steps only or in the last epoch
            if render_frames and (self.train_steps % 10 == 0 or self.train_steps == self.epochs - 1):
                # T x H x W x C
                sequence = np.array(render_frames)
                # T x C x H x W
                sequence = np.transpose(sequence, [0, 3, 1, 2])
                # B x T x C x H x W
                sequence = np.expand_dims(sequence, axis=0)
                tag = 'end-training' if self.train_steps == self.epochs - 1 else 'training'
                self.logger.add_video(tag, vid_tensor=sequence, global_step=self.train_steps, fps=8)

    def play(self):
        pass
