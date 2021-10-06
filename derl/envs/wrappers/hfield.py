import math
import sys

import gym
import numpy as np

from derl.config import cfg
from derl.utils import exception as exu
from derl.utils import geom as gu
from derl.utils import mjpy as mu
from derl.utils import spaces as spu

np.set_printoptions(threshold=sys.maxsize)

"""Observation Wrappers."""


class HfieldObs1D(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.divs = cfg.HFIELD.NUM_DIVS
        behind, front, _, _ = [_ * self.divs for _ in cfg.HFIELD.OBS_SIZE]
        self.observation_space = spu.update_obs_space(
            env, {"hfield": (front + behind,)}
        )

    def observation(self, obs):
        hfield = self.metadata["hfield"]
        behind, front, _, _ = [_ * self.divs for _ in cfg.HFIELD.OBS_SIZE]
        x_pos, y_pos, _ = self.unwrapped.sim.data.get_body_xpos("torso/0")
        _, col_idx = obs["hfield_idx"]

        c_min = col_idx - behind
        c_max = col_idx + front

        obs["hfield"] = hfield[0, c_min:c_max]

        obs["hfield"] = obs["hfield"] - hfield[0, col_idx]
        obs["hfield"] = obs["hfield"].flatten()
        return obs


class HfieldObs2D(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.divs = cfg.HFIELD.NUM_DIVS

        if cfg.HFIELD.ADAPTIVE_OBS:
            hfield_bounds = self._cal_hfield_bounds(env.metadata["orig_height"])
        else:
            hfield_bounds = [int(_ * self.divs) for _ in cfg.HFIELD.OBS_SIZE]

        self.mask_row, self.mask_col = self._create_hfield_mask(hfield_bounds)
        self.observation_space = spu.update_obs_space(
            env, {"hfield": (self.mask_row.size,)}
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._add_hfield_obs(obs, None)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self._add_hfield_obs(obs, info)
        return obs, rew, done, info

    def _add_hfield_obs(self, obs, info):
        sim = self.unwrapped.sim
        hfield = self.metadata["hfield"]
        x_pos, y_pos, _ = sim.data.get_body_xpos("torso/0")

        # Get un-rotated mask
        mask_row, mask_col = self.mask_row, self.mask_col

        # Maybe rotate the mask. Original x-axis of torso frame was (1, 0, 0).
        # Get current torso x-axis and rotate mask accordingly
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        x_dir = gu.normalize_vec(torso_frame[:, 0][:2])
        if info:
            rot_angle = gu.angle_between([1, 0], x_dir)
            mask_row, mask_col = self._rotate_mask(rot_angle)

        # Translate the mask
        row_idx, col_idx = obs["hfield_idx"]
        mask_row = mask_row + row_idx
        mask_col = mask_col + col_idx

        corner_points = self._get_corner_points(mask_row, mask_col)
        self.metadata["corner_points"] = corner_points
        if cfg.HFIELD.VIZ:
            self._debug_viz(corner_points)

        try:
            obs["hfield"] = hfield[mask_row, mask_col]
        except IndexError as e:
            uid = self.metadata["unimal_id"]
            exu.handle_exception(
                e, "ERROR in Hfield: {}".format(uid), unimal_id=uid
            )

        obs["hfield"] = obs["hfield"] - hfield[row_idx, col_idx]
        obs["hfield"] = obs["hfield"].flatten()
        return obs

    def _sample_non_uniform(self, end):
        """Sample points with reducing density from 0 to end."""
        # Scale between [0, 1)
        points = np.asarray(list(range(0, end))) / end
        # Refer: https://en.wikipedia.org/wiki/Inverse_transform_sampling#Examples
        points = -1 * np.log(1 - points)
        # Scale between [0, num]
        points = (points / points[-1]) * end
        # Keep unique points
        points = np.unique(points.astype(int))
        return points

    def _sample_non_uniform_range(self, low, hi):
        """Sample points with reducing density from from origin between low and hi."""
        hi_points = self._sample_non_uniform(hi)
        if low > 0:
            low_points = np.flip(-1 * self._sample_non_uniform(low))
            points = np.concatenate((low_points, hi_points))
        else:
            points = hi_points
        return np.unique(points)

    def _create_hfield_mask(self, hfield_bounds):
        behind, front, right, left = hfield_bounds
        x = self._sample_non_uniform_range(behind, front)
        y = self._sample_non_uniform_range(left, right)
        col, row = np.meshgrid(x, y)
        return row, col

    def _rotate_mask(self, angle):
        """Orient the mask in the direction unimal is heading."""
        x, y = self.mask_col, self.mask_row
        xr = np.cos(angle) * x + np.sin(angle) * y
        yr = -np.sin(angle) * x + np.cos(angle) * y

        row_rot = np.around(yr).astype(int)
        col_rot = np.around(xr).astype(int)
        return row_rot, col_rot

    def _cal_hfield_bounds(self, unimal_height):
        # Height and hfield obs size pair. e.g. If unimal height is h1 then
        # unimal can see in front, left and right till b1.
        h1, b1, h2, b2 = cfg.HFIELD.ADAPTIVE_OBS_SIZE

        # Linear fit corresponding to above two points
        slope = (b2 - b1) / (h2 - h1)

        obs_size = (unimal_height - h1) * slope + b1
        obs_size = int(math.ceil(obs_size * cfg.HFIELD.NUM_DIVS))

        # behind = min(10, obs_size)
        return [obs_size] * 4

    def _get_corner_points(self, mask_row, mask_col):
        # front_left, front_right, back_left, back_right
        corner_points = [(0, -1), (-1, -1), (0, 0), (-1, 0)]
        name_cp_pos = {}
        names = ["fl", "fr", "bl", "br"]
        self.hfield_markers = []

        terrain_module = "Terrain"
        if cfg.ENV.TASK == "escape_bowl":
            terrain_module = "Bowl"

        for name, point in zip(names, corner_points):
            row, col = mask_row[point], mask_col[point]
            x_pos, y_pos = self.unwrapped.modules[terrain_module].idx_to_pos(
                [row, col]
            )
            name_cp_pos[name] = [x_pos, y_pos]
        return name_cp_pos

    def _debug_viz(self, name_cp_pos):
        """Viz the hfield input while rendering."""
        for name, corner_point in name_cp_pos.items():
            cp = corner_point + [3]
            self.hfield_markers.append(
                {
                    "label": name,
                    "size": np.array([0.2, 0.2, 0.01]),
                    "rgba": np.array([0, 0, 1, 0.4]),
                    "pos": np.array(cp),
                }
            )

        # Update viewer with markers, if any
        if self.unwrapped.viewer is not None:
            for marker in self.hfield_markers:
                self.unwrapped.viewer.add_marker(**marker)


class UnimalHeightObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spu.update_obs_space(env, {"torso_height": (1,)})

    def observation(self, obs):
        x_pos, y_pos, z_pos = self.sim.data.get_body_xpos("torso/0")

        # Height of ground is constant if Floor
        if "Floor" in cfg.ENV.MODULES:
            obs["torso_height"] = round(z_pos - 1, 3)
            return obs

        row_idx, col_idx = obs["hfield_idx"]
        try:
            terrain_z = self.metadata["hfield"][row_idx, col_idx]
            # In case of gap terrain_z will be negative, clip at 0
            if cfg.ENV.TASK not in ["incline", "push_box_incline"]:
                terrain_z = max(0, terrain_z)
        except IndexError as e:
            uid = self.metadata["unimal_id"]
            exu.handle_exception(
                e, "ERROR in Hfield: {}".format(uid), unimal_id=uid
            )
        obs["torso_height"] = round(z_pos - terrain_z, 3)
        return obs


"""Reward Wrappers."""


class AvoidWallReward(gym.RewardWrapper):
    def reward(self, reward):
        if check_agent_wall_contact(self.unwrapped.sim):
            reward = reward - cfg.ENV.AVOID_REWARD_WEIGHT
        return reward


class StandReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_height = env.metadata["orig_height"]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        stand_reward = (
            min(obs["torso_height"], 2.0 * self.orig_height)
            * cfg.ENV.STAND_REWARD_WEIGHT
        )
        info["__reward__stand"] = stand_reward
        rew += stand_reward
        return obs, rew, done, info


class ExploreTerrainReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        l, w, _ = cfg.TERRAIN.SIZE
        self.divs = cfg.HFIELD.NUM_DIVS
        self.visit_grid = np.zeros((w * 2, l * 2))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.visit_grid[:, :] = 0.0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        prev_visit_count = np.sum(self.visit_grid)
        row_idx, col_idx = obs["placement_idx"]
        try:
            row_idx = int(math.floor(row_idx/self.divs))
            col_idx = int(math.floor(col_idx/self.divs))
            self.visit_grid[row_idx, col_idx] = 1.0
        except IndexError:
            pass
        curr_visit_count = np.sum(self.visit_grid)
        explore_reward = curr_visit_count - prev_visit_count
        info["__reward__explore"] = explore_reward
        info["metric"] = curr_visit_count
        rew += explore_reward
        return obs, rew, done, info


"""Termination Wrappers."""


class TerminateOnWallContact(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if check_agent_wall_contact(self.unwrapped.sim):
            done = True
            rew = rew - cfg.ENV.AVOID_REWARD_WEIGHT
        return obs, rew, done, info


class TerminateOnTerrainEdge(gym.Wrapper):
    """Terminate episode if unimals are near edge of terrain (along y dir)."""

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.is_near_edge():
            done = True
        return obs, rew, done, info

    def is_near_edge(self):
        x_pos, y_pos, _ = self.sim.data.get_body_xpos("torso/0")
        if cfg.TERRAIN.SIZE[1] - abs(y_pos) <= 1:
            return True
        else:
            return False


class TerminateOnFalling(gym.Wrapper):
    """Teriminate episode if torso falls below a certain height."""

    def __init__(self, env):
        super().__init__(env)
        self.fall_threshold = env.metadata["fall_threshold"]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.has_fallen(obs):
            done = True
        return obs, rew, done, info

    def has_fallen(self, obs):
        if obs["torso_height"] <= self.fall_threshold:
            return True
        else:
            return False


class TerminateOnRotation(gym.Wrapper):
    """Teriminate episode if unimal does cartwheel!"""

    def __init__(self, env):
        super().__init__(env)
        self.sum = 0
        self.count = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.is_rotating(obs):
            done = True

        if done or (self.step_count == self.spec.max_episode_steps):
            self.sum = 0
            self.count = 0
        return obs, rew, done, info

    def is_rotating(self, obs):
        # Get torso subtreeangmom sensordata
        angular_velocity = self.sim.data.sensordata[9:12].copy()
        self.sum += np.linalg.norm(angular_velocity)
        self.count += 1
        avg = round(self.sum / self.count, 2)
        # Empirically, determined value
        if avg >= 35:
            return True
        else:
            return False


class TerminateOnEscape(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # Due to computational reasons we will never have hfield obs size more
        # than 5 (diag would be ~7, we keep some buffer and make it 8)
        if info["metric"] >= cfg.TERRAIN.SIZE[0] - 8:
            done = True
            rew += 100
        return obs, rew, done, info


def check_agent_wall_contact(sim):
    contact_geoms = mu.get_active_contacts(sim)
    contact_geoms = [
        name for contact in contact_geoms for name in contact if "wall" in name
    ]
    return contact_geoms
