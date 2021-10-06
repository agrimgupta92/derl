import numpy as np
from lxml import etree

from derl.config import cfg
from derl.utils import mjpy as mu
from derl.utils import placement as plu
from derl.utils import sample as su
from derl.utils import xml as xu


class Objects:
    """Generate objects and add them to the env."""

    def __init__(self, random_state=None):

        self.np_random = random_state

        self.len, self.width, self.h = cfg.TERRAIN.SIZE
        self.divs = cfg.HFIELD.NUM_DIVS
        self.placement_grid = np.zeros(
            (
                self.width * cfg.HFIELD.NUM_DIVS * 2,
                self.len * cfg.HFIELD.NUM_DIVS * 2,
            )
        )

        self.obj_pos = None
        self.goal_pos = None
        # Sequence of goals for goal follower
        self.curr_goal_idx = -1
        # Sites which should be visible when rendering
        self.object_sites = []
        self.markers = []

    def place_object(self, env, size, pos, center=None, modify_hfield=False):
        size_x, size_y, size_z = size
        os = np.asarray([size_x, size_y])
        if not pos:
            pos = plu.place_on_grid(self, os, center=center)

        if not pos:
            return None
        else:
            x, y = pos

        row_idx, col_idx = env.modules["Terrain"].pos_to_idx([x, y])
        os_in_divs = np.ceil(os * self.divs).astype(int)
        max_height = np.max(
            env.metadata["hfield"][
                row_idx - os_in_divs[1] : row_idx + os_in_divs[1],
                col_idx - os_in_divs[0] : col_idx + os_in_divs[0],
            ]
        )
        pos = [x, y, size_z + max_height + 0.1]

        if modify_hfield:
            env.metadata["hfield"][
                row_idx - os_in_divs[1] : row_idx + os_in_divs[1],
                col_idx - os_in_divs[0] : col_idx + os_in_divs[0],
            ] = size_z + max_height

        return pos

    def add_box(self, env, idx, material="self"):
        bs = cfg.OBJECT.BOX_SIDE
        pos = self.place_object(
            env, [bs] * 3, cfg.OBJECT.BOX_POS, center=self.pos_to_grid_idx([0, 0])
        )
        self.obj_pos = pos
        name = "box/{}".format(idx)
        box = xu.body_elem(name, pos)
        # Add joint for box
        box.append(xu.joint_elem("{}/root".format(name), "free", "free"))
        # Add box geom
        box.append(
            etree.Element(
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "size": xu.arr2str([bs, bs, bs]),
                    "condim": "3",
                    "mass": str(cfg.OBJECT.BOX_MASS),
                    "material": material,
                },
            )
        )
        # Add site at each face center
        face_sites = [
            [bs, 0, 0],
            [-bs, 0, 0],
            [0, bs, 0],
            [0, -bs, 0],
            [0, 0, bs],
            [0, 0, -bs],
        ]
        for i, fs in enumerate(face_sites):
            site_name = "box/face/{}/{}".format(i, idx)
            box.append(xu.site_elem(site_name, fs, "box_face_site"))
            self.object_sites.append(site_name)
        return box

    def add_ball(self, env, idx):
        bs = cfg.OBJECT.BALL_RADIUS
        pos = self.place_object(
            env, [bs] * 3, None, center=self.pos_to_grid_idx([0, 0])
        )
        self.obj_pos = pos
        name = "ball/{}".format(idx)
        ball = xu.body_elem(name, pos)
        # Add joint for ball
        ball.append(xu.joint_elem("{}/root".format(name), "free", "free"))
        # Add ball geom
        # Solref values taken from dm_control, explation:
        # http://www.mujoco.org/book/modeling.html (restitution section)
        ball.append(
            etree.Element(
                "geom",
                {
                    "name": name,
                    "type": "sphere",
                    "size": str(bs),
                    "condim": "6",
                    "density": "600.0",
                    "material": "ball",
                    "priority": "1",
                    "friction": "0.7 0.005 0.005",
                    "solref": "-10000 -30"
                },
            )
        )
        return ball

    def add_goal(self, env):
        gs = 0.5
        gh = 0.01
        center = None
        if self.obj_pos is not None:
            center = self.pos_to_grid_idx(self.obj_pos[:2])
        self.goal_pos = self.place_object(
            env,
            [gs, gs, gh],
            cfg.OBJECT.GOAL_POS,
            center=center
        )
        self.markers.append(
            {
                "label": "",
                "size": np.array([gs, gs, gh]),
                "rgba": np.array([1, 0, 0, 0.4]),
                "pos": np.array(self.goal_pos),
            }
        )

    def add_obstacles(self, env):
        obstacles = []
        for idx in range(cfg.OBJECT.NUM_OBSTACLES):
            bl = su.sample_from_range(
                cfg.OBJECT.OBSTACLE_LEN_RANGE, self.np_random
            )
            bw = su.sample_from_range(
                cfg.OBJECT.OBSTACLE_WIDTH_RANGE, self.np_random
            )
            obstacle_dims = [bl, bw, 2]
            pos = self.place_object(
                env, obstacle_dims, None, modify_hfield=True
            )
            if not pos:
                continue
            name = "obstacle/{}".format(idx)
            box = xu.body_elem(name, pos)
            # Add box geom
            box.append(
                etree.Element(
                    "geom",
                    {
                        "name": name,
                        "type": "box",
                        "size": xu.arr2str(obstacle_dims),
                        "condim": "3",
                        "mass": "1.0",
                        "material": "self",
                    },
                )
            )
            obstacles.append(box)
        return obstacles

    def modify_xml_step(self, env, root, tree):
        worldbody = root.findall("./worldbody")[0]
        xml_elems = []

        # Add box if task is manipulation
        if cfg.ENV.TASK in ["manipulation", "push_box_incline"]:
            if cfg.OBJECT.TYPE == "box":
                xml_elems.append(self.add_box(env, 1))
            else:
                xml_elems.append(self.add_ball(env, 1))

        if cfg.ENV.TASK in ["manipulation", "point_nav", "push_box_incline"]:
            self.add_goal(env)

        if cfg.ENV.TASK == "obstacle":
            xml_elems.extend(self.add_obstacles(env))

        for elem in xml_elems:
            worldbody.append(elem)

        env.metadata["markers"] = self.markers
        env.metadata["object_sites"] = self.object_sites

        # xu.save_etree_as_xml(tree, "1.xml")
        # np.save("outfile.npy", env.metadata["hfield"])

    def modify_sim_step(self, env, sim):
        self.obj_qpos_idxs = np.array(
            mu.qpos_idxs_from_joint_prefix(sim, cfg.OBJECT.TYPE)
        )
        self.obj_qvel_idxs = np.array(
            mu.qvel_idxs_from_joint_prefix(sim, cfg.OBJECT.TYPE)
        )

    def observation_step(self, env, sim):
        if cfg.ENV.TASK in ["manipulation", "push_box_incline"]:
            return self.manipulation_obs_step(env, sim)
        elif cfg.ENV.TASK in ["point_nav"]:
            return self.point_nav_obs_step(env, sim)
        elif cfg.ENV.TASK in ["exploration"]:
            return self.exploration_obs_step(env, sim)
        elif cfg.ENV.TASK == "obstacle":
            return {}
        else:
            raise ValueError("Task not supported: {}".format(cfg.ENV.TASK))

    def manipulation_obs_step(self, env, sim):
        pos = sim.data.qpos.flat.copy()
        vel = sim.data.qvel.flat.copy()

        obj_pos = pos[self.obj_qpos_idxs][:3]
        obj_vel = vel[self.obj_qvel_idxs][:3]

        obj_rot_vel = vel[self.obj_qvel_idxs][3:]

        # Convert obj pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")
        agent_qvel_idxs = env.modules["Agent"].agent_qvel_idxs
        agent_vel = vel[agent_qvel_idxs][:3]

        obj_rel_pos = obj_pos - torso_pos
        obj_rel_vel = obj_vel - agent_vel

        goal_rel_pos = self.goal_pos - torso_pos
        obj_state = np.vstack(
            (obj_rel_pos, obj_rel_vel, obj_rot_vel, goal_rel_pos)
        )
        obj_state = obj_state.dot(torso_frame).ravel()
        return {"obj": obj_state}

    def point_nav_obs_step(self, env, sim):
        # Convert box pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")

        goal_rel_pos = self.goal_pos - torso_pos
        goal_state = goal_rel_pos.dot(torso_frame).ravel()
        return {"goal": goal_state}

    def exploration_obs_step(self, env, sim):
        x_pos, y_pos, _ = sim.data.get_body_xpos("torso/0")
        row_idx, col_idx = self.pos_to_grid_idx([x_pos, y_pos])
        return {"placement_idx": np.asarray([row_idx, col_idx])}

    def grid_idx_to_pos(self, idx):
        """Convert from grid --> mujoco pos."""
        idx = idx / self.divs
        row_idx, col_idx = idx
        x_pos = col_idx - cfg.TERRAIN.START_FLAT
        y_pos = row_idx - self.width
        pos = [x_pos, y_pos]
        pos = [round(_, 2) for _ in pos]
        return pos

    def pos_to_grid_idx(self, pos):
        """Convert from mujoco pos to grid."""
        x_pos, y_pos = pos
        row_idx = y_pos + self.width
        col_idx = x_pos + cfg.TERRAIN.START_FLAT
        idx = [row_idx * self.divs, col_idx * self.divs]
        idx = [int(_) for _ in idx]
        return idx

    def assert_cfg(self):
        pass
