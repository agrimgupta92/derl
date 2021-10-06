import gym
import numpy as np

from derl.config import cfg


class ReachReward(gym.Wrapper):
    """Reach reward used in PointNav task."""

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        agent_goal_d_before = np.linalg.norm(
            info["goal_pos"] - info["xy_pos_before"]
        )
        agent_goal_d_after = info["agent_goal_d"]

        scale_rew, time_penalty, success_rew = cfg.ENV.REACH_REWARD_PARAMS
        reach_reward = (agent_goal_d_before - agent_goal_d_after) * scale_rew
        reach_reward += time_penalty
        if agent_goal_d_after <= cfg.OBJECT.SUCCESS_MARGIN:
            reach_reward += success_rew
            info["__reward__success"] = 1
        info["__reward__reach"] = reach_reward
        rew += reach_reward
        return obs, rew, done, info
