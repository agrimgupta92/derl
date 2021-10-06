from collections import OrderedDict

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.utils import seeding

import derl.utils.exception as exu
from derl.config import cfg
from derl.utils import spaces as spu
from derl.utils import xml as xu

DEFAULT_SIZE = 1024
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class UnimalEnv(gym.Env):
    """Superclass for all Unimal tasks."""

    def __init__(self, xml_str, unimal_id):
        self.frame_skip = 4

        self.viewer = None
        self._viewers = {}

        self.xml_str = xml_str
        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "unimal_id": unimal_id,
            "markers": [],
        }
        self.observation_space = None
        self.module_classes = []
        self.modules = OrderedDict()
        self.seed()

    def add_module(self, cname):
        self.module_classes.append(cname)

    def _init_modules(self):
        self.modules = OrderedDict()
        for cname in self.module_classes:
            module = cname()
            name_str = module.__class__.__name__
            self.modules[name_str] = module
            self.modules[name_str].np_random = self.np_random

    def _get_obs(self):
        obs = {}
        for _, module in self.modules.items():
            obs.update(module.observation_step(self, self.sim))
        return obs

    def _get_sim(self):
        root, tree = xu.etree_from_xml(self.xml_str, ispath=False)
        self._init_modules()
        # Modify the xml
        for _, module in self.modules.items():
            module.modify_xml_step(self, root, tree)

        xml_str = xu.etree_to_str(root)
        model = mujoco_py.load_model_from_xml(xml_str)
        sim = mujoco_py.MjSim(model)
        # Update module fields which require sim
        for _, module in self.modules.items():
            module.modify_sim_step(self, sim)
        return sim

    ###########################################################################
    # Functions to setup env attributes
    ###########################################################################

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip

    def _set_action_space(self):
        bounds = self.sim.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def seed(self, seed=None):
        if isinstance(seed, list):
            seed = seed[0]
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ###########################################################################
    # Common rewards and costs
    ###########################################################################
    def control_cost(self, action):
        control_cost = cfg.ENV.CTRL_COST_WEIGHT * np.sum(np.square(action))
        return control_cost

    def calculate_energy(self):
        if cfg.HFIELD.DIM == 1:
            torque = self.sim.data.qfrc_actuator[3:]
        else:
            torque = self.sim.data.qfrc_actuator[1:]
        return int(np.sum(np.absolute(torque)))

    ###########################################################################
    # Reset sim
    ###########################################################################
    def reset(self):
        self.sim = self._get_sim()
        self.step_count = 0
        if self.viewer is not None:
            self.viewer.update_sim(self.sim)
        obs = self.reset_model()
        self._set_action_space()
        # This is just a temp initialization, the final obs space will depend
        # on SelectKeysWrapper
        self.observation_space = spu.convert_obs_to_space(obs)
        self.metadata["video.frames_per_second"] = int(np.round(1.0 / self.dt))
        return obs

    def reset_model(self):
        noise_low = -cfg.ENV.RESET_NOISE_SCALE
        noise_high = cfg.ENV.RESET_NOISE_SCALE

        init_qpos = self.sim.data.qpos.ravel().copy()
        init_qvel = self.sim.data.qvel.ravel().copy()

        qpos_idx = self.modules["Agent"].agent_qpos_idxs
        qvel_idx = self.modules["Agent"].agent_qvel_idxs

        init_qpos[qpos_idx] = init_qpos[qpos_idx] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=len(qpos_idx)
        )
        init_qvel[qvel_idx] = init_qvel[qvel_idx] + self.np_random.uniform(
            low=noise_low, high=noise_high, size=len(qvel_idx)
        )

        self.set_state(init_qpos, init_qvel)

        observation = self._get_obs()
        return observation

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (
            self.sim.model.nv,
        )
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    ###########################################################################
    # MjStep
    ###########################################################################

    def do_simulation(self, ctrl):
        self.step_count += 1
        self.sim.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            try:
                self.sim.step()
            except Exception as e:
                uid = self.metadata["unimal_id"]
                exu.handle_exception(
                    e, "ERROR in MjStep: {}".format(uid), unimal_id=uid
                )

    ###########################################################################
    # Viewing and rendering methods
    ###########################################################################

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "side"

            if (
                camera_id is None
                and camera_name in self.sim.model._camera_name2id
            ):
                camera_id = self.sim.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[
                1
            ]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}
