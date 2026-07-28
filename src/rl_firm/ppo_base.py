import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from .env import BaselineABMEnv, make_custom_env

@dataclass
class HyperParameters:
    total_timesteps: int = 1000000 # Total timesteps of the run
    learning_rate: float = 0.0003  # The learning rate of the optimizer
    num_envs: int = 8              # The number of parallel ABM environments
    num_steps: int = 1024          # The number of the steps to run for each envivonrment per policy rollout
    anneal_lr: bool = False        # Toggle learning rate annealing for policy and value networks
    gamma: float = 0.999           # The discount factor gamma
    gae: bool = True               # Toogle using general advantage estimation (GAE)
    gae_lambda: float = 0.95       # The lambda for the GAE
    num_minibatches: int = 4       # The number of mini-batches per batch
    update_epochs: int = 10        # The number of epochs when optimizing
    norm_adv: bool = True          # Toggle advantages normalization
    clip_coef: float = 0.2         # The surrogate clipping coefficient
    clip_vloss: bool = False       # Toggle whether or not to use a clipped loss for the value function
    ent_coef: float = 0.0          # The coefficient of the entropy
    vf_coef: float = 0.5           # The coefficient of the value function
    max_grad_norm: float = 0.5     # The maximum norm for the gradient clipping
    target_kl: float = None        # The target KL divergence threshold

    @property
    def batch_size(self) -> int:
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        return int(self.batch_size // self.num_minibatches)

    @property
    def num_iterations(self) -> int:
        return int(self.total_timesteps // self.batch_size)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # Shared feature network
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
        self.cat_head = layer_init(nn.Linear(64, 3), std=0.01)
        self.cont_mean = layer_init(nn.Linear(64, 2), std=0.01)
        self.cont_logstd = nn.Parameter(torch.zeros(1, 2))

    def get_value(self, x):
        return self.critic(self.network(x))

    def _get_probs(self, x):
        hidden = self.network(x)

        cat_logits = self.cat_head(hidden)
        cont_mean = self.cont_mean(hidden)
        cont_logstd = self.cont_logstd.expand_as(cont_mean)
        cont_std = torch.exp(cont_logstd)

        cat_probs = Categorical(logits=cat_logits)
        cont_probs = Normal(cont_mean, cont_std)

        return hidden, cat_probs, cont_probs

    def get_action_and_value(self, x, action=None):
        hidden, cat_probs, cont_probs = self._get_probs(x)
        
        if action is None:
            cat_action = cat_probs.sample()
            cont_action = cont_probs.sample()
            action = {
                'hr_action': cat_action,
                'price_adjustment': cont_action[..., 0:1],
                'wage_adjustment': cont_action[..., 1:2]
            }
        else:
            cat_action = action['hr_action']
            cont_action = torch.cat([action['price_adjustment'], action['wage_adjustment']], dim=-1)
            
        return (
            action,
            cont_probs.log_prob(cont_action).sum(1) + cat_probs.log_prob(cat_action),
            cat_probs.entropy() + cont_probs.entropy().sum(1),
            self.critic(hidden)
        )

    @torch.no_grad()
    def predict(self, x, deterministic=False):
        _, cat_probs, cont_probs = self._get_probs(x)
        if deterministic:
            cat_action = cat_probs.logits.argmax(-1)
            cont_action = cont_probs.mean
        else:
            cat_action = cat_probs.sample()
            cont_action = cont_probs.sample()
        return {
            'hr_action': cat_action,
            'price_adjustment': cont_action[..., 0:1],
            'wage_adjustment': cont_action[..., 1:2]
        }


def train(hp: HyperParameters, run_name: str):
    writer = SummaryWriter(f"../runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(hp).items()])),
    )
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f'Training on device {device}')
    envs = gym.vector.AsyncVectorEnv([make_custom_env() for _ in range(hp.num_envs)])
    
    agent = PPOAgent(envs).to(device)   
    optimizer = torch.optim.Adam(agent.parameters(), lr=hp.learning_rate, eps=1e-5)

    # ALGO logic: storage setup
    obs = torch.zeros((hp.num_steps, hp.num_envs) + envs.single_observation_space.shape).to(device)
    hr_actions = torch.zeros((hp.num_steps, hp.num_envs) + envs.single_action_space['hr_action'].shape).to(device)
    price_actions = torch.zeros((hp.num_steps, hp.num_envs) + envs.single_action_space['price_adjustment'].shape).to(device)
    wage_actions = torch.zeros((hp.num_steps, hp.num_envs) + envs.single_action_space['wage_adjustment'].shape).to(device)
    log_probs = torch.zeros((hp.num_steps, hp.num_envs)).to(device)
    rewards = torch.zeros((hp.num_steps, hp.num_envs)).to(device)
    dones = torch.zeros((hp.num_steps, hp.num_envs)).to(device)
    values = torch.zeros((hp.num_steps, hp.num_envs)).to(device)

    # LOG returns
    episode_rewards = []
    episode_lengths = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_ob = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(hp.num_envs).to(device)
    num_updates = hp.total_timesteps // hp.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if hp.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * hp.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, hp.num_steps):
            global_step += 1 * hp.num_envs
            obs[step] = next_ob
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_ob)
                values[step] = value.flatten()

            hr_actions[step] = action['hr_action']
            price_actions[step] = action['price_adjustment']
            wage_actions[step] = action['wage_adjustment']
            log_probs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, infos = envs.step({
                'hr_action': action['hr_action'].cpu().numpy(),
                'price_adjustment': action['price_adjustment'].cpu().numpy(),
                'wage_adjustment': action['wage_adjustment'].cpu().numpy()
            })
            rewards[step] = torch.tensor(reward.astype(np.float32)).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            if "episode" in infos:
                for idx, done in enumerate(infos["_episode"]):
                    if done:
                        episode_lengths.append(infos["episode"]["l"][idx])
                        episode_rewards.append(infos['episode']['r'][idx])
                        recent_avg_length = np.mean(episode_lengths[-20:])
                        recent_avg = np.mean(episode_rewards[-20:])
                        print(f"Episode {len(episode_rewards)}: Recent average episodic reward: {recent_avg:.2f}, recent average episodic length: {recent_avg_length:.2f}. Global step: {global_step}")
                        writer.add_scalar(
                            "charts/episodic_reward",
                            infos["episode"]["r"][idx],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            infos["episode"]["l"][idx],
                            global_step,
                        )
        # bootstrap reward if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if hp.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(hp.num_steps)):
                    if t == hp.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + hp.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + hp.gamma * hp.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(hp.num_steps)):
                    if t == hp.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + hp.gamma * nextnonterminal * next_return
                advantages = returns - values
                
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_hr_actions = hr_actions.reshape((-1,) + envs.single_action_space['hr_action'].shape)
        b_price_actions = price_actions.reshape((-1,) + envs.single_action_space['price_adjustment'].shape)
        b_wage_actions = wage_actions.reshape((-1,) + envs.single_action_space['wage_adjustment'].shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimizing the policy and value network
        b_inds = np.arange(hp.batch_size)
        clipfracs = []
        for epoch in range(hp.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, hp.batch_size, hp.minibatch_size):
                end = start + hp.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_inds],
                    {
                        'hr_action': b_hr_actions[mb_inds],
                        'price_adjustment': b_price_actions[mb_inds],
                        'wage_adjustment': b_wage_actions[mb_inds]
                    }
                )
                logratio = newlogprob - b_log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > hp.clip_coef).float().mean().cpu()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if hp.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - hp.clip_coef, 1 + hp.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = new_values.view(-1)
                if hp.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -hp.clip_coef, hp.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - hp.ent_coef * entropy_loss + v_loss * hp.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), hp.max_grad_norm)
                optimizer.step()

            if hp.target_kl is not None:
                if approx_kl > hp.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
    return agent