# algos/mappo_agent.py
import torch
import torch.nn as nn
import numpy as np
from algos.networks import SharedActor, CentralizedCritic


class MAPPOManager:
    def __init__(self, obs_dim, state_dim, act_dim=2, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.base_lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef

        # Master networks on CUDA GPU
        self.actor = SharedActor(obs_dim, act_dim).to(self.device)
        self.critic = CentralizedCritic(state_dim).to(self.device)

        # High-speed CPU inference mirrors (zero CUDA driver launch latency for single-step rollouts)
        self.actor_cpu = SharedActor(obs_dim, act_dim).to("cpu")
        self.critic_cpu = CentralizedCritic(state_dim).to("cpu")
        self.sync_cpu_actor()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

    def sync_cpu_actor(self):
        """Synchronizes GPU network weights and running normalization buffers to CPU mirror."""
        self.actor_cpu.load_state_dict(self.actor.state_dict())
        self.critic_cpu.load_state_dict(self.critic.state_dict())

    def anneal_lr(self, current_step, total_steps):
        frac = max(0.1, 1.0 - 0.9 * (current_step / float(total_steps)))
        new_lr = self.base_lr * frac
        for pg in self.actor_opt.param_groups:
            pg["lr"] = new_lr
        for pg in self.critic_opt.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def compute_gae(self, rewards, values, dones, next_value):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, obs_b, state_b, act_b, logprob_b, val_b, adv_b, ret_b, epochs=10, batch_size=512):
        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)
        state_t = torch.tensor(state_b, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_b, dtype=torch.float32, device=self.device)
        old_logprob_t = torch.tensor(logprob_b, dtype=torch.float32, device=self.device)
        old_val_t = torch.tensor(val_b, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv_b, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret_b, dtype=torch.float32, device=self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        num_samples = obs_t.shape[0]
        indices = np.arange(num_samples)

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                mb = indices[start:start + batch_size]

                _, new_logprob, entropy = self.actor.get_action_and_logprob(obs_t[mb], act_t[mb])
                ratio = torch.exp(new_logprob - old_logprob_t[mb])

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                new_val = self.critic(state_t[mb])
                val_clipped = old_val_t[mb] + torch.clamp(new_val - old_val_t[mb], -self.clip_coef, self.clip_coef)
                v_loss1 = (new_val - ret_t[mb]) ** 2
                v_loss2 = (val_clipped - ret_t[mb]) ** 2
                critic_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        # Keep CPU mirror in sync after GPU update
        self.sync_cpu_actor()

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }
