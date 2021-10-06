import numpy as np
from gym import utils

from derl.config import cfg
from derl.envs.modules.agent import Agent
from derl.envs.modules.floor import Floor
from derl.envs.modules.patrol_goals import PatrolGoals
from derl.envs.tasks.unimal import UnimalEnv
from derl.envs.wrappers.hfield import HfieldObs2D
from derl.envs.wrappers.hfield import StandReward
from derl.envs.wrappers.hfield import TerminateOnFalling
from derl.envs.wrappers.hfield import UnimalHeightObs
from derl.envs.wrappers.metrics import PatrolMetric
from derl.envs.wrappers.reach import ReachReward


class PatrolTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id):
        UnimalEnv.__init__(self, xml_str, unimal_id)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        ctrl_cost = self.control_cost(action)
        reward = -ctrl_cost
        observation = self._get_obs()

        # Calculate distance between agent and goal
        goal_pos = self.modules["PatrolGoals"].goal_pos[:2]
        agent_goal_d = np.linalg.norm(goal_pos - xy_pos_after)

        toggle = 0
        if agent_goal_d <= cfg.OBJECT.SUCCESS_MARGIN:
            self.modules["PatrolGoals"].toggle_goal(self)
            toggle = 1

        info = {
            "x_pos": xy_pos_after[0] - cfg.TERRAIN.SIZE[0],
            "y_pos": xy_pos_after[1],
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "agent_goal_d": agent_goal_d,
            "goal_pos": goal_pos,
            "__reward__energy": self.calculate_energy(),
            "__reward__ctrl": ctrl_cost,
            "toggle": toggle
        }
        # Update viewer with markers, if any
        if self.viewer is not None:
            self.viewer._markers[:] = []
            for marker in self.metadata["markers"]:
                self.viewer.add_marker(**marker)

        return observation, reward, False, info


def make_env_patrol(xml, unimal_id):
    env = PatrolTask(xml, unimal_id)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = ReachReward(env)
    env = PatrolMetric(env)
    return env
