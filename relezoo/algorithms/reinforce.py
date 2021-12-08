from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from gym import Env


class Policy:
    def __init__(self, net: nn.Module, learning_rate: float = 1e-2):
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)

    def act(self, obs: torch.Tensor) -> (torch.Tensor, int):
        logits = self._get_policy(obs)
        action = logits.sample().item()
        return action

    def _get_policy(self, obs: torch.Tensor):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)

    def learn(self, batch_obs: torch.Tensor, batch_actions: torch.Tensor, batch_weights: torch.Tensor):
        self.optimizer.zero_grad()
        batch_loss = self._compute_loss(batch_obs, batch_actions, batch_weights)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss

    def _compute_loss(self, obs, actions, weights):
        logp = self._get_policy(obs).log_prob(actions)
        return -(logp * weights).mean()

    def save(self, save_path: str):
        torch.save(self.net, save_path)


class Reinforce:

    def __init__(self, env: Env, policy: Optional[Policy] = None, logger: Optional[SummaryWriter] = None):
        self.env = env
        self.obs_space = env.observation_space.shape[0]
        self.act_space = env.action_space.n
        self.policy = policy
        self.logger = logger
        self.train_steps = 0

    def train(self, epochs: int = 50, batch_size: int = 5000, render: bool = False):
        assert self.policy is not None, "The policy is not defined."

        with tqdm(total=epochs) as progress:
            for i in range(1, epochs + 1):
                is_last_epoch = i == epochs
                batch_loss, batch_returns, batch_lens = self._train_epoch(batch_size, render, is_last_epoch)
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

    def _train_epoch(self,
                     batch_size: int = 5000,
                     render: bool = False,
                     is_last_epoch: bool = False) -> (float, float, int):
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
            if render and not render_epoch:
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

                if len(batch_obs) > batch_size:
                    break

        batch_loss = self.policy.learn(
            torch.from_numpy(np.array(batch_obs)),
            torch.from_numpy(np.array(batch_actions)),
            torch.from_numpy(np.array(batch_weights))
        )

        self._log(is_last_epoch, batch_loss, batch_returns, batch_lens, render_frames)

        self.train_steps += 1

        return batch_loss, batch_returns, batch_lens

    def _log(self, is_last_epoch: bool, batch_loss, batch_returns, batch_lens, render_frames):
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
            if render_frames and (self.train_steps % 10 == 0 or is_last_epoch):
                # T x H x W x C
                sequence = np.array(render_frames)
                # T x C x H x W
                sequence = np.transpose(sequence, [0, 3, 1, 2])
                # B x T x C x H x W
                sequence = np.expand_dims(sequence, axis=0)
                tag = 'end-training' if is_last_epoch else 'training'
                self.logger.add_video(tag, vid_tensor=sequence, global_step=self.train_steps, fps=8)

    def save(self, save_path: str):
        self.policy.save(save_path)

    def load(self, load_path: str):
        net = torch.load(load_path)
        self.policy = Policy(net)

    def play(self, episodes: int) -> (float, int):
        assert self.policy is not None, "The policy is not defined."
        with tqdm(total=episodes) as progress:
            ep_rewards = []
            ep_lengths = []
            for i in range(1, episodes + 1):
                obs = self.env.reset()
                ep_length = 1
                ep_reward = 0
                while True:
                    action = self.policy.act(obs)
                    obs, reward, done, _ = self.env.step(action)
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
