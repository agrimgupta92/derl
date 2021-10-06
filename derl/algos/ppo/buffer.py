import gym
import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from derl.config import cfg


class Buffer(object):
    def __init__(self, obs_space, act_shape):
        T, P = cfg.PPO.TIMESTEPS, cfg.PPO.NUM_ENVS

        if isinstance(obs_space, gym.spaces.Dict):
            self.obs = {}
            for obs_type, obs_space_ in obs_space.spaces.items():
                self.obs[obs_type] = torch.zeros(T, P, *obs_space_.shape)
        else:
            self.obs = torch.zeros(T, P, *obs_space.shape)

        self.act = torch.zeros(T, P, *act_shape)
        self.val = torch.zeros(T, P, 1)
        self.rew = torch.zeros(T, P, 1)
        self.ret = torch.zeros(T, P, 1)
        self.logp = torch.zeros(T, P, 1)
        self.masks = torch.ones(T, P, 1)
        self.timeout = torch.ones(T, P, 1)

        self.step = 0

    def insert(self, obs, act, logp, val, rew, masks, timeouts):
        if isinstance(obs, dict):
            for obs_type, obs_val in obs.items():
                self.obs[obs_type][self.step] = obs_val
        else:
            self.obs[self.step] = obs
        self.act[self.step] = act
        self.val[self.step] = val
        self.rew[self.step] = rew
        self.logp[self.step] = logp
        self.masks[self.step] = masks
        self.timeout[self.step] = timeouts

        self.step = (self.step + 1) % cfg.PPO.TIMESTEPS

    def compute_returns(self, next_value):
        """
        We use ret as approximate gt for value function for training. When step
        is terminal state we need to handle two cases:
        1. Agent Died: timeout[step] = 1 and mask[step] = 0. This ensures
           gae is reset to 0 and self.ret[step] = 0.
        2. Agent Alive but done true due to timeout: timeout[step] = 0
           mask[step] = 0. This ensures gae = 0 and self.ret[step] = val[step].
        """
        gamma, gae_lambda = cfg.PPO.GAMMA, cfg.PPO.GAE_LAMBDA
        # val: (T+1, P, 1), self.val: (T, P, 1) next_value: (P, 1)
        val = torch.cat((self.val.squeeze(), next_value.t())).unsqueeze(2)
        gae = 0
        for step in reversed(range(cfg.PPO.TIMESTEPS)):
            delta = (
                self.rew[step]
                + gamma * val[step + 1] * self.masks[step]
                - val[step]
            ) * self.timeout[step]
            gae = delta + gamma * gae_lambda * self.masks[step] * gae
            self.ret[step] = gae + val[step]

    def get_sampler(self, adv):
        dset_size = cfg.PPO.TIMESTEPS * cfg.PPO.NUM_ENVS

        assert dset_size >= cfg.PPO.BATCH_SIZE

        sampler = BatchSampler(
            SubsetRandomSampler(range(dset_size)),
            cfg.PPO.BATCH_SIZE,
            drop_last=True,
        )

        for idxs in sampler:
            batch = {}
            batch["ret"] = self.ret.view(-1, 1)[idxs]

            if isinstance(self.obs, dict):
                batch["obs"] = {}
                for ot, ov in self.obs.items():
                    batch["obs"][ot] = ov.view(-1, *ov.size()[2:])[idxs]
            else:
                batch["obs"] = self.obs.view(-1, *self.obs.size()[2:])[idxs]

            batch["val"] = self.val.view(-1, 1)[idxs]
            batch["act"] = self.act.view(-1, self.act.size(-1))[idxs]
            batch["adv"] = adv.view(-1, 1)[idxs]
            batch["logp_old"] = self.logp.view(-1, 1)[idxs]
            yield batch
