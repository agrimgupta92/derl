import gym
import numpy as np

from .running_mean_std import RunningMeanStd
from .vec_env import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        training=True,
    ):
        VecEnvWrapper.__init__(self, venv)
        obs_space = self.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            shape = obs_space["proprioceptive"].shape
        else:
            shape = obs_space.shape
        self.ob_rms = RunningMeanStd(shape=shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret[news] = 0.0
        return obs, rews, news, infos

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if isinstance(obs, dict):
                obs_p = obs["proprioceptive"]
            else:
                obs_p = obs

            if self.training and update:
                self.ob_rms.update(obs_p)

            obs_p = np.clip(
                (obs_p - self.ob_rms.mean)
                / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            if isinstance(obs, dict):
                obs["proprioceptive"] = obs_p
            else:
                obs = obs_p
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
