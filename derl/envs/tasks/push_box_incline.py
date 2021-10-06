import numpy as np
from gym import utils
from scipy.spatial import distance as scipy_distance

from derl.config import cfg
from derl.envs.modules.agent import Agent
from derl.envs.modules.objects import Objects
from derl.envs.modules.terrain import Terrain
from derl.envs.tasks.unimal import UnimalEnv
from derl.envs.wrappers.hfield import StandReward
from derl.envs.wrappers.hfield import TerminateOnFalling
from derl.envs.wrappers.hfield import UnimalHeightObs


class PushBoxIncline(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id):
        UnimalEnv.__init__(self, xml_str, unimal_id)
        self.obj_type = cfg.OBJECT.TYPE
        self.obj_name = "{}/1".format(self.obj_type)
        self.angle = np.deg2rad(abs(cfg.TERRAIN.INCLINE_ANGLE))

    def _cal_agent_obj_dist(self):
        agent_pos = [
            self.sim.data.get_site_xpos(agent_site).copy()
            for agent_site in self.metadata["agent_sites"]
        ]
        obj_pos = [
            self.sim.data.get_site_xpos(obj_site).copy()
            for obj_site in self.metadata["object_sites"]
        ]
        distance = scipy_distance.cdist(agent_pos, obj_pos, "euclidean")
        return np.min(distance)

    ###########################################################################
    # Sim step and reset
    ###########################################################################
    def step(self, action):
        agent_obj_d_before = self._cal_agent_obj_dist()
        obj_pos_before = self.sim.data.get_body_xpos(self.obj_name)[:2].copy()
        self.do_simulation(action)

        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        agent_obj_d_after = self._cal_agent_obj_dist()
        obj_pos_after = self.sim.data.get_body_xpos(self.obj_name)[:2].copy()

        # Reward given to agent to reach/or be near to obj
        reach_reward = (agent_obj_d_before - agent_obj_d_after) * 100.0
        if (
            agent_obj_d_after <= cfg.OBJECT.SUCCESS_MARGIN
            and not self.reached_obj
        ):
            reach_reward += 10.0
            self.reached_obj = True

        # Reward given to agent to "push" obj to be near to goal
        push_reward = 0.0
        goal_pos = self.modules["Objects"].goal_pos[:2]
        obj_goal_d_after = None
        if self.reached_obj:
            obj_goal_d_before = np.linalg.norm(goal_pos - obj_pos_before)
            obj_goal_d_after = np.linalg.norm(goal_pos - obj_pos_after)
            push_reward = (obj_goal_d_before - obj_goal_d_after) * 100.0
            if obj_goal_d_after <= cfg.OBJECT.SUCCESS_MARGIN:
                push_reward += 10.0
                self.reach_goal_obj = True

        ctrl_cost = self.control_cost(action)
        reward = reach_reward + push_reward - ctrl_cost
        observation = self._get_obs()

        # Metric is distance travelled by box
        x_dist_box = np.linalg.norm(
            np.asarray(cfg.OBJECT.BOX_POS) - obj_pos_after
        ) / np.cos(self.angle)

        info = {
            "x_pos": xy_pos_after[0],
            "y_pos": xy_pos_after[1],
            "__reward__reach": reach_reward,
            "__reward__push": push_reward,
            "__reward__energy": self.calculate_energy(),
            "__reward__ctrl": ctrl_cost,
            "__reward__manipulation": reach_reward + push_reward,
            "metric": x_dist_box,
        }

        # Update viewer with markers, if any
        if self.viewer is not None:
            self.viewer._markers[:] = []
            for marker in self.metadata["markers"]:
                self.viewer.add_marker(**marker)

        return observation, reward, False, info

    def reset(self):
        obs = super().reset()
        self.reached_obj = False
        self.reach_goal_agent = False
        self.reach_goal_obj = False
        return obs


def make_env_push_box_incline(xml, unimal_id):
    env = PushBoxIncline(xml, unimal_id)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    return env
