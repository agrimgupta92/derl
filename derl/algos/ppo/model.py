import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from derl.utils import model as tu  # torch

from .obs_encoder import ObsEncoder


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_feat_dim, act_dim):
        super(MLPGaussianActor, self).__init__()

        self.mu_net = tu.w_init(nn.Linear(obs_feat_dim, act_dim))
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs_feat):
        mu = self.mu_net(obs_feat)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space):
        super(ActorCritic, self).__init__()

        self.v_obs_feat = ObsEncoder(obs_space)
        self.pi_obs_feat = ObsEncoder(obs_space)

        assert self.v_obs_feat.obs_feat_dim is not None
        obs_feat_dim = self.v_obs_feat.obs_feat_dim
        self.v = MLPCritic(obs_feat_dim)
        self.pi = MLPGaussianActor(obs_feat_dim, action_space.shape[0])

    def forward(self, obs, act=None):
        v_obs_feat = self.v_obs_feat(obs)
        val = self.v(v_obs_feat)

        pi_obs_feat = self.pi_obs_feat(obs)
        pi = self.pi(pi_obs_feat)
        if act is not None:
            logp = pi.log_prob(act).sum(-1, keepdim=True)
            entropy = pi.entropy().mean()
            return val, pi, logp, entropy
        else:
            return val, pi, None, None


class MLPCritic(nn.Module):
    def __init__(self, obs_feat_dim):
        super(MLPCritic, self).__init__()

        self.critic = tu.w_init(nn.Linear(obs_feat_dim, 1))

    def forward(self, obs_feat):
        return self.critic(obs_feat)


# This slight awk sepration between actor critic and agent is due to DDP in
# pytorch. We can't have (I don't know how) these functions as part of the
# ActorCritic class.
class Agent:
    def __init__(self, actor_critic):
        self.ac = actor_critic

    @torch.no_grad()
    def act(self, obs):
        val, pi, _, _ = self.ac(obs)
        act = pi.sample()
        logp = pi.log_prob(act).sum(-1, keepdim=True)
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, _, _, _ = self.ac(obs)
        return val
