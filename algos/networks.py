# algos/networks.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningMeanStd(nn.Module):
    """Tracks running mean and variance of tensors for online MAPPO input/value normalization."""
    def __init__(self, shape=(), epsilon=1e-5):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def normalize(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


def layer_init(layer, std=1.4142, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedActor(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden_dim=64):
        super().__init__()
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

    def get_action_and_logprob(self, obs, action=None, deterministic=False, update_norm=False):
        if update_norm:
            self.obs_norm.update(obs)
        norm_obs = self.obs_norm.normalize(obs)
        mean = self.net(norm_obs)
        if deterministic:
            return mean, None, None
        std = torch.exp(torch.clamp(self.actor_logstd, -2.0, 0.5)).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            raw_action = dist.sample()
            action = torch.clamp(raw_action, -1.0, 1.0)
        else:
            action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.state_norm = RunningMeanStd(shape=(state_dim,))
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, global_state, update_norm=False):
        if update_norm:
            self.state_norm.update(global_state)
        norm_state = self.state_norm.normalize(global_state)
        return self.net(norm_state).squeeze(-1)
