from gym import utils

from derl.config import cfg
from derl.envs.modules.agent import Agent
from derl.envs.modules.objects import Objects
from derl.envs.modules.terrain import Terrain
from derl.envs.tasks.unimal import UnimalEnv
from derl.envs.wrappers.hfield import HfieldObs2D
from derl.envs.wrappers.hfield import StandReward
from derl.envs.wrappers.hfield import TerminateOnFalling
from derl.envs.wrappers.hfield import TerminateOnTerrainEdge
from derl.envs.wrappers.hfield import UnimalHeightObs


class ObstacleTask(UnimalEnv, utils.EzPickle):
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
        x_vel, y_vel = xy_vel

        forward_reward = cfg.ENV.FORWARD_REWARD_WEIGHT * x_vel

        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        observation = self._get_obs()

        info = {
            "__reward__ctrl": ctrl_cost,
            "__reward__energy": self.calculate_energy(),
            "x_pos": xy_pos_after[0],
            "x_vel": x_vel,
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "__reward__forward": forward_reward,
            "metric": xy_pos_after[0],
        }

        return observation, reward, False, info


def make_env_obstacle(xml, unimal_id):
    env = ObstacleTask(xml, unimal_id)
    # Add modules
    for module in cfg.ENV.MODULES:
        env.add_module(globals()[module])
    # Reset is needed to setup observation spaces, sim etc which might be
    # needed by wrappers
    env.reset()
    # Add all wrappers
    env = UnimalHeightObs(env)
    env = StandReward(env)
    env = TerminateOnFalling(env)
    env = HfieldObs2D(env)
    env = TerminateOnTerrainEdge(env)
    return env
