from gym import utils

from derl.config import cfg
from derl.envs.modules.agent import Agent
from derl.envs.modules.floor import Floor
from derl.envs.modules.terrain import Terrain
from derl.envs.tasks.unimal import UnimalEnv
from derl.envs.wrappers.hfield import AvoidWallReward
from derl.envs.wrappers.hfield import HfieldObs1D
from derl.envs.wrappers.hfield import HfieldObs2D
from derl.envs.wrappers.hfield import StandReward
from derl.envs.wrappers.hfield import TerminateOnFalling
from derl.envs.wrappers.hfield import TerminateOnRotation
from derl.envs.wrappers.hfield import TerminateOnTerrainEdge
from derl.envs.wrappers.hfield import TerminateOnWallContact
from derl.envs.wrappers.hfield import UnimalHeightObs


class LocomotionTask(UnimalEnv, utils.EzPickle):
    def __init__(self, xml_str, unimal_id):
        UnimalEnv.__init__(self, xml_str, unimal_id)

    ###########################################################################
    # Sim step and reset
    ###########################################################################

    def step(self, action):
        if cfg.HFIELD.DIM == 1:
            info, forward_reward = self.step_1d(action)
        else:
            info, forward_reward = self.step_2d(action)

        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        observation = self._get_obs()

        info_ = {
            "__reward__ctrl": ctrl_cost,
            "__reward__energy": self.calculate_energy(),
        }
        info.update(info_)

        return observation, reward, False, info

    def step_1d(self, action):
        x_pos_before = self.sim.data.qpos[0]
        self.do_simulation(action)
        x_pos_after = self.sim.data.qpos[0]
        x_vel = (x_pos_after - x_pos_before) / self.dt
        forward_reward = cfg.ENV.FORWARD_REWARD_WEIGHT * x_vel
        pos_info = {
            "x_pos": x_pos_after,
            "x_vel": x_vel,
            "__reward__forward": forward_reward,
        }
        return pos_info, forward_reward

    def step_2d(self, action):
        xy_pos_before = self.sim.data.get_body_xpos("torso/0")[:2].copy()
        self.do_simulation(action)
        xy_pos_after = self.sim.data.get_body_xpos("torso/0")[:2].copy()

        xy_vel = (xy_pos_after - xy_pos_before) / self.dt
        x_vel, y_vel = xy_vel

        forward_reward = cfg.ENV.FORWARD_REWARD_WEIGHT * x_vel
        pos_info = {
            "x_pos": xy_pos_after[0],
            "x_vel": x_vel,
            "xy_pos_before": xy_pos_before,
            "xy_pos_after": xy_pos_after,
            "__reward__forward": forward_reward,
            "metric": xy_pos_after[0],
        }
        return pos_info, forward_reward


def make_env_locomotion(xml, unimal_id):
    env = LocomotionTask(xml, unimal_id)
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
    if "Terrain" in cfg.ENV.MODULES:
        if cfg.HFIELD.DIM == 1:
            env = HfieldObs1D(env)
        else:
            env = HfieldObs2D(env)
            env = TerminateOnTerrainEdge(env)

        if "AvoidWallReward" in cfg.ENV.WRAPPERS:
            env = AvoidWallReward(env)
        if "TerminateOnWallContact" in cfg.ENV.WRAPPERS:
            env = TerminateOnWallContact(env)
        if "TerminateOnRotation" in cfg.ENV.WRAPPERS:
            # Keep in the bottom, as resetting count and sum depends on done
            # value
            env = TerminateOnRotation(env)
    return env
