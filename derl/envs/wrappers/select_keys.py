from collections import OrderedDict

import gym
import numpy as np
from gym.spaces import Box
from gym.spaces import Dict


class SelectKeysWrapper(gym.ObservationWrapper):
    """Select obs keys as input to model."""

    def __init__(self, env, keys_to_keep):
        super().__init__(env)
        self.keys_to_keep = keys_to_keep
        inf = np.float32(np.inf)
        observation_space = {
            k: Box(-inf, inf, v.shape, np.float32)
            for k, v in self.observation_space.spaces.items()
            if k in self.keys_to_keep
        }
        self.observation_space = Dict(OrderedDict(observation_space))

    def observation(self, observation):
        obs = {k: v for k, v in observation.items() if k in self.keys_to_keep}
        return obs
