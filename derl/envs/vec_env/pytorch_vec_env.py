import torch

from .vec_env import VecEnvWrapper


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = self._obs_np2torch(obs)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = self._obs_np2torch(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        return obs, reward, done, info

    def _obs_np2torch(self, obs):
        if isinstance(obs, dict):
            for ot, ov in obs.items():
                obs[ot] = torch.from_numpy(obs[ot]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs
