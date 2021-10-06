import numpy as np
from gym import utils

from derl.config import cfg
from derl.envs.modules.agent import Agent
from derl.envs.modules.objects import Objects
from derl.envs.modules.terrain import Terrain
from derl.envs.tasks.unimal import UnimalEnv
from derl.envs.wrappers.hfield import ExploreTerrainReward
from derl.envs.wrappers.hfield import HfieldObs2D
from derl.envs.wrappers.hfield import StandReward
from derl.envs.wrappers.hfield import TerminateOnFalling
from derl.envs.wrappers.hfield import UnimalHeightObs


class ExplorationTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id):
        UnimalEnv.__init__(self, xml_str, unimal_id)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        xy_vel = (xy_pos_after - xy_pos_before) / self.dt
        forward_reward = cfg.ENV.FORWARD_REWARD_WEIGHT * np.linalg.norm(xy_vel)

        ctrl_cost = self.control_cost(action)
        reward = -ctrl_cost + forward_reward
        observation = self._get_obs()

        info = {
            "x_pos": xy_pos_after[0] - cfg.TERRAIN.SIZE[0],
            "y_pos": xy_pos_after[1],
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "__reward__forward": forward_reward,
            "__reward__energy": self.calculate_energy(),
            "__reward__ctrl": ctrl_cost,
        }

        # Update viewer with markers, if any
        if self.viewer is not None:
            self.viewer._markers[:] = []
            for marker in self.metadata["markers"]:
                self.viewer.add_marker(**marker)

        return observation, reward, False, info


def make_env_exploration(xml, unimal_id):
    env = ExplorationTask(xml, unimal_id)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = ExploreTerrainReward(env)
    return env
