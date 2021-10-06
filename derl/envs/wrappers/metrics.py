import gym
import numpy as np

from derl.config import cfg


class ReachMetric(gym.Wrapper):
    """Calculate human interpretable metric for PointNav task."""

    def __init__(self, env):
        super().__init__(env)
        self.time_after_success = 0

    def is_done(self, done):
        if done or (self.step_count == self.spec.max_episode_steps):
            return True
        else:
            return False

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if not self.is_done(done):
            if "__reward__success" in info:
                self.time_after_success += 1
            return obs, rew, done, info

        orig_dist_from_goal = np.linalg.norm(
            np.asarray(info["goal_pos"]) - np.asarray([cfg.TERRAIN.SIZE[0], 0])
        )

        # Distance component of metric, 1 when agent is at goal.
        metric_d = 1 - (info["agent_goal_d"] / orig_dist_from_goal)
        # How fast agent reached the goal
        metric_f = self.time_after_success / 1000
        info["metric"] = metric_d + metric_f

        # Reset time tracker
        self.time_after_success = 0

        return obs, rew, done, info


class ManipulationMetric(gym.Wrapper):
    """Calculate human interpretable metric for Manipulation task."""

    def __init__(self, env):
        super().__init__(env)
        self.time_after_success = 0

    def is_done(self, done):
        if done or (self.step_count == self.spec.max_episode_steps):
            return True
        else:
            return False

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if not self.is_done(done):
            if info["reach_goal_obj"] and info["reach_goal_agent"]:
                self.time_after_success += 1
            return obs, rew, done, info

        metric_r = 1.0
        metric_ga = 1.0
        metric_gb = 1.0
        metric_f = 0.0

        if not info["reached_obj"]:
            orig_dist = np.linalg.norm(info["goal_pos"])
            info["metric"] = 1 - (info["agent_obj_d_after"] / orig_dist)
            return obs, rew, done, info

        if not info["reach_goal_agent"]:
            metric_ga = 1 - (info["agent_goal_d_after"] / info["init_obj_goal_d"])

        if not info["reach_goal_obj"]:
            metric_gb = 1 - (info["obj_goal_d_after"] / info["init_obj_goal_d"])

        if info["reach_goal_obj"] and info["reach_goal_agent"]:
            # How fast agent reached the goal
            metric_f = self.time_after_success / 1000

        info["metric"] = metric_r + metric_ga + metric_gb + metric_f
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.time_after_success = 0
        return obs


class PatrolMetric(gym.Wrapper):
    """Calculate human interpretable metric for Patrol task."""

    def __init__(self, env):
        super().__init__(env)
        self.total_toggles = 0

    def is_done(self, done):
        if done or (self.step_count == self.spec.max_episode_steps):
            return True
        else:
            return False

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if not self.is_done(done):
            self.total_toggles += info["toggle"]
            return obs, rew, done, info

        # Distance component of metric, 1 when agent is at goal.
        metric_d = 1 - (info["agent_goal_d"] / (cfg.TERRAIN.PATROL_HALF_LEN * 2))

        info["metric"] = metric_d + self.total_toggles
        # Reset toggle tracker
        self.total_toggles = 0

        return obs, rew, done, info
