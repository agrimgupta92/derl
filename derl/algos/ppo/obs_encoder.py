import gym
import torch
import torch.nn as nn

from derl.config import cfg
from derl.utils import model as tu


class ObsEncoder(nn.Module):
    """Encode all observations into a single vector."""

    def __init__(self, obs_space):
        super(ObsEncoder, self).__init__()

        if isinstance(obs_space, gym.spaces.Dict):
            if len(obs_space.spaces) == 1:
                self.obs_encoder = MLPObsEncoder(
                    obs_space["proprioceptive"].shape[0]
                )
            else:
                self.obs_encoder = ConcatMLPEncoder(obs_space)
        else:
            self.obs_encoder = MLPObsEncoder(obs_space.shape[0])

        self.obs_feat_dim = self.obs_encoder.obs_feat_dim

    def forward(self, obs):
        return self.obs_encoder(obs)


class MLPObsEncoder(nn.Module):
    """Encoder when only proprioceptive features are in obs."""

    def __init__(self, obs_dim):
        super(MLPObsEncoder, self).__init__()

        mlp_dims = [obs_dim] + cfg.MODEL.PRO_HIDDEN_DIMS
        self.encoder = tu.make_mlp(mlp_dims)
        self.obs_feat_dim = cfg.MODEL.PRO_HIDDEN_DIMS[-1]

    def forward(self, obs):
        if isinstance(obs, dict):
            return self.encoder(obs["proprioceptive"])
        else:
            return self.encoder(obs)


class ConcatMLPEncoder(nn.Module):
    def __init__(self, obs_space):
        super(ConcatMLPEncoder, self).__init__()

        assert isinstance(obs_space, gym.spaces.Dict)
        assert len(obs_space.spaces) != 1

        self.encoders = nn.ModuleDict()

        concat_dims = 0
        for obs_type, space in obs_space.spaces.items():
            mlp_dims = [space.shape[0]] + cfg.MODEL.PRO_HIDDEN_DIMS
            self.encoders[obs_type] = tu.make_mlp(mlp_dims)
            concat_dims += cfg.MODEL.PRO_HIDDEN_DIMS[-1]

        mlp_dims = [concat_dims] + cfg.MODEL.OBS_FEAT_HIDDEN_DIMS
        self.encode_concat = tu.make_mlp(mlp_dims)
        self.obs_feat_dim = cfg.MODEL.OBS_FEAT_HIDDEN_DIMS[-1]

    def forward(self, obs):
        encoded_obs = []
        for ot, ov in obs.items():
            encoded_obs.append(self.encoders[ot](ov))

        # concat along obs_feat dim
        concat_obs_feat = torch.cat(encoded_obs, dim=1)
        obs_feat = self.encode_concat(concat_obs_feat)
        return obs_feat
