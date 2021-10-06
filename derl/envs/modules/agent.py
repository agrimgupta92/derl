import numpy as np
from lxml import etree

import derl.utils.camera as cu
import derl.utils.xml as xu
from derl.config import cfg
from derl.utils import mjpy as mu


class Agent:
    def __init__(self, random_state=None):

        self.np_random = random_state

    def modify_xml_step(self, env, root, tree):
        # Store agent height
        worldbody = root.findall("./worldbody")[0]
        head = xu.find_elem(worldbody, "body", "name", "torso/0")[0]
        pos = xu.str2arr(head.get("pos"))
        orig_height = pos[2] - cfg.TERRAIN.SIZE[2]
        # Don't subtract offset for pre-built walkers
        if not cfg.ENV.WALKER:
            orig_height -= cfg.BODY.FLOOR_OFFSET

        # Change position of agent to center for some tasks
        if cfg.ENV.TASK in ["point_nav", "exploration"]:
            pos[0] = cfg.TERRAIN.SIZE[0]
            head.set("pos", xu.arr2str(pos))
        if cfg.ENV.TASK == "patrol":
            pos[0] = -cfg.TERRAIN.PATROL_HALF_LEN
            head.set("pos", xu.arr2str(pos))
        if cfg.ENV.TASK in ["incline", "push_box_incline"]:
            angle = np.deg2rad(abs(cfg.TERRAIN.INCLINE_ANGLE))
            pos[0] = np.cos(angle) * (-cfg.TERRAIN.SIZE[0] + 2.0)
            height = np.sin(angle) * (cfg.TERRAIN.SIZE[0] - 2.0)
            if cfg.TERRAIN.INCLINE_ANGLE > 0:
                pos[2] += height
            else:
                pos[2] -= height
            head.set("pos", xu.arr2str(pos))
            head.set("euler", xu.arr2str([0, cfg.TERRAIN.INCLINE_ANGLE, 0]))

        # The center position in escape_bowl is ~0 height so subtract the terrain
        # height which is added in merge agent with base
        if cfg.ENV.TASK == "escape_bowl":
            pos[2] = pos[2] - cfg.TERRAIN.SIZE[2]
            head.set("pos", xu.arr2str(pos))

        self._add_cameras(head)
        self._add_fixed_cameras(worldbody)

        env.metadata["orig_height"] = round(orig_height, 2)
        env.metadata["fall_threshold"] = orig_height * cfg.ENV.STAND_HEIGHT_RATIO

    def modify_sim_step(self, env, sim):
        self.agent_qpos_idxs = np.array(mu.qpos_idxs_for_agent(sim))
        self.agent_qvel_idxs = np.array(mu.qvel_idxs_for_agent(sim))

        site_prefixes = ["limb/btm/", "limb/mid/", "torso"]
        env.metadata["agent_sites"] = mu.names_from_prefixes(
            sim, site_prefixes, "site"
        )

        self.limb_btm_sites = [
            site for site in env.metadata["agent_sites"] if "limb/btm" in site
        ]

    def observation_step(self, env, sim):
        observations = []
        for obs_type in cfg.ENV.OBS_TYPES:
            func = getattr(self, obs_type)
            observations.append(func(sim))

        return {
            "proprioceptive": np.concatenate(observations).ravel(),
        }

    def _add_fixed_cameras(self, worldbody):
        cameras = [
            cu.PATROL_VIEW,
            cu.TUNE_CAMERA,
        ]
        insert_pos = 1

        for spec in cameras:
            worldbody.insert(insert_pos, xu.camera_elem(spec))
            insert_pos += 1

    def _add_cameras(self, head):
        cameras = [
            cu.INCLINE_VIEW,
            cu.MANI_VIEW,
            cu.OBSTACLE_VIEW,
            cu.FT_VIEW,
            cu.VT_VIEW,
            cu.LEFT_VIEW,
            cu.TOP_DOWN,
            cu.FRONT_VIEW,
            cu.REAR_VIEW,
        ]
        insert_pos = 0
        for idx, child_elem in enumerate(head):
            if child_elem.tag == "camera":
                insert_pos = idx + 1
                break

        for spec in cameras:
            head.insert(insert_pos, xu.camera_elem(spec))
            insert_pos += 1

    ###########################################################################
    # Proprioceptive observations
    ###########################################################################
    def position(self, sim):
        pos = sim.data.qpos.flat.copy()
        pos = pos[self.agent_qpos_idxs]

        if not cfg.ENV.SKIP_SELF_POS:
            return pos
        # Ignores horizontal position to maintain translational invariance
        if cfg.HFIELD.DIM == 1:
            pos = pos[1:]
        else:
            # Skip the 7 DoFs of the free root joint
            pos = pos[7:]
        return pos

    def velocity(self, sim):
        vel = sim.data.qvel.flat.copy()
        return vel[self.agent_qvel_idxs]

    def imu_vel(self, sim):
        # Return torso acceleration, torso gyroscope and torso velocity
        return sim.data.sensordata[:9].copy()

    def touch(self, sim):
        # Return scalar force, each limb/torso has one touch sensor
        return sim.data.sensordata[12:].copy()

    def extremities(self, sim):
        """Returns limb positions in torso/0 frame."""
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")
        positions = []
        for site_name in self.limb_btm_sites:
            torso_to_limb = sim.data.get_site_xpos(site_name) - torso_pos
            positions.append(torso_to_limb.dot(torso_frame)[:2])
        extremities = np.hstack(positions)

        return extremities


def merge_agent_with_base(agent, ispath=True):
    base_xml = cfg.UNIMAL_TEMPLATE
    root_b, tree_b = xu.etree_from_xml(base_xml)
    root_a, tree_a = xu.etree_from_xml(agent, ispath=ispath)

    worldbody = root_b.findall("./worldbody")[0]
    agent_body = xu.find_elem(root_a, "body", "name", "torso/0")[0]

    # Update agent z pos based on starting terrain
    pos = xu.str2arr(agent_body.get("pos"))
    pos[2] += cfg.TERRAIN.SIZE[2]
    agent_body.set("pos", xu.arr2str(pos))
    worldbody.append(agent_body)

    actuator_a = root_a.findall("./actuator")[0]
    actuator_b = root_b.findall("./actuator")[0]
    agent_motors = xu.find_elem(actuator_a, "motor")
    actuator_b.extend(agent_motors)

    sensor_a = root_a.findall("./sensor")[0]
    sensor_b = root_b.findall("./sensor")[0]
    sensor_b.extend(list(sensor_a))
    return xu.etree_to_str(root_b)


def extract_agent_from_xml(xml_path):
    root, tree = xu.etree_from_xml(xml_path)
    agent = etree.Element("agent", {"model": "unimal"})
    unimal = xu.find_elem(root, "body", "name", "torso/0")[0]
    actuator = root.findall("./actuator")[0]
    sensor = root.findall("./sensor")[0]
    agent.append(unimal)
    agent.append(actuator)
    agent.append(sensor)
    agent = xu.etree_to_str(agent)
    return agent
