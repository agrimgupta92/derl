import os
import random
from collections import defaultdict

import imageio
import numpy as np
from lxml import etree

from derl.config import cfg
from derl.utils import file as fu
from derl.utils import geom as gu
from derl.utils import mjpy as mu
from derl.utils import sample as su
from derl.utils import xml as xu

HEAD = "torso/0"


class SymmetricUnimal:
    """Representation for symmetric unimal."""

    def __init__(self, id_, init_path=None):
        self.id = id_

        if init_path:
            self._init_from_state(init_path)
            self.parent_id = fu.path2id(init_path)
        else:
            self._init_new_unimal()
            self.parent_id = ""

        self.mutation_ops = cfg.EVO.MUTATION_OPS
        self.worldbody.append(self.unimal)
        # Mirror sites: If you add a limb in one mirror site you have to also
        # add the same limb to it's mirror counterpart.
        if self.init_state:
            self.mirror_sites = self.init_state["mirror_sites"]
        else:
            self.mirror_sites = {}

    def _init_new_unimal(self):
        """Construct a new unimal with just head."""
        self.init_state = None
        # Initalize xml from template
        self.root, self.tree = xu.etree_from_xml(cfg.UNIMAL_TEMPLATE)
        self.worldbody = self.root.findall("./worldbody")[0]
        self.actuator = self.root.findall("./actuator")[0]
        self.contact = self.root.findall("./contact")[0]

        # Set of contact pairs. Each item is a list (geom_name, geom_name).
        # Note: We cannot store ids as there is not gurantee they will be the
        # same.
        self.contact_pairs = set()

        self.num_limbs = 0
        # In case of delete_limb op limb_idx can differ from num_limbs
        self.limb_idx = 0
        # Unimals starts with a head.
        self.num_torso = 1

        # List of geom names. e.g [[limb1], [limb2, limb3], [limb4]] where
        # limb2 and limb3 are symmetric counterparts.
        self.limb_list = []

        # List of torso names
        self.torso_list = [0]

        # List of torso from where limbs can grow
        self.growth_torso = [0]

        # Body params
        self.body_params = {
            "torso_mode": random.choice(["horizontal_y", "vertical"]),
            "torso_density": su.sample_from_range(cfg.TORSO.DENSITY_RANGE),
            "limb_density": su.sample_from_range(cfg.LIMB.DENSITY_RANGE),
            "num_torso": su.sample_from_range(cfg.TORSO.NUM_TORSO_RANGE),
        }

        # Contains information about a limb like: orientation, parent etc
        self.limb_metadata = defaultdict(dict)

        # Construct unimal
        self.unimal = self._construct_head()

    def _init_from_state(self, init_path):
        self.init_state = fu.load_pickle(init_path)

        self.root, self.tree = xu.etree_from_xml(self.init_state["xml_path"])
        self.worldbody = self.root.findall("./worldbody")[0]
        self.actuator = self.root.findall("./actuator")[0]
        self.contact = self.root.findall("./contact")[0]

        self.contact_pairs = self.init_state["contact_pairs"]
        self.num_limbs = self.init_state["num_limbs"]
        self.limb_idx = self.init_state["limb_idx"]
        self.num_torso = self.init_state["num_torso"]
        self.limb_list = self.init_state["limb_list"]
        self.body_params = self.init_state["body_params"]
        self.limb_metadata = self.init_state["limb_metadata"]
        self.torso_list = self.init_state["torso_list"]
        self.growth_torso = self.init_state["growth_torso"]
        self.unimal = xu.find_elem(
            self.root, "body", attr_type="name", attr_value=HEAD
        )[0]

        self._remove_permanent_contacts()

    def _construct_head(self):
        # Placeholder position which will be updated based final unimal height
        head = xu.body_elem("torso/0", [0, 0, 0])
        # Add joints between unimal and the env (generally the floor)
        if cfg.HFIELD.DIM == 1:
            head.append(xu.joint_elem("rootx", "slide", "free", axis=[1, 0, 0]))
            head.append(xu.joint_elem("rootz", "slide", "free", axis=[0, 0, 1]))
            head.append(xu.joint_elem("rooty", "hinge", "free", axis=[0, 1, 0]))
        else:
            head.append(xu.joint_elem("root", "free", "free"))

        head.append(xu.site_elem("root", None, "imu_vel"))

        head_params = self._choose_torso_params()
        r = head_params["torso_radius"]
        # Add the actual head
        head.append(
            etree.Element(
                "geom",
                {
                    "name": HEAD,
                    "type": "sphere",
                    "size": "{}".format(r),
                    "condim": str(cfg.TORSO.CONDIM),
                    "density": str(head_params["torso_density"]),
                },
            )
        )
        # Add cameras
        head.append(
            etree.fromstring(
                '<camera name="side" pos="0 -7 2" xyaxes="1 0 0 0 1 2" mode="trackcom"/>'
            )
        )

        # Add site where limb can be attached
        head.append(xu.site_elem("torso/0", [0, 0, 0], "growth_site"))
        # Add btm pos site
        head.append(xu.site_elem("torso/btm_pos/0", [0, 0, -r], "btm_pos_site"))
        # Add site for touch sensor
        head.append(
            xu.site_elem("torso/touch/0", None, "touch_site", str(r + 0.01))
        )
        # Add torso growth sites
        if "horizontal" in self.body_params["torso_mode"]:
            head.append(
                xu.site_elem(
                    "torso/{}/0".format(self.body_params["torso_mode"]),
                    [-r, 0, 0],
                    "torso_growth_site",
                )
            )
        else:
            head.append(
                xu.site_elem("torso/vertical/0", [0, 0, -r], "torso_growth_site",)
            )
        return head

    def _construct_limb(self, idx, site, site_type, limb_params, orient=None):
        # Get parent radius
        parent_idx = xu.name2id(site)
        if "torso" in site.get("name"):
            parent_name = "torso/{}".format(parent_idx)
        else:
            parent_name = "limb/{}".format(parent_idx)
        p_r = self._get_limb_r(parent_name)

        # Get limb_params
        r, h = limb_params["limb_radius"], limb_params["limb_height"]

        # Get limb orientation
        if orient:
            h, theta, phi = orient
        else:
            h, theta, phi = su.sample_orient(limb_params["limb_height"])
            # If we are growing two limbs, ensure that the random orientation
            # is less than 180. Prevents duplication of unimals where the
            # unimal only differs in which orientation was choosen first.
            if theta > np.pi and site_type == "mirror_growth_site":
                theta = 2 * np.pi - theta
                orient = (h, theta, phi)
            orient = (h, theta, phi)

        # Set limb pos
        pos = xu.str2arr(site.get("pos"))
        pos = xu.add_list(pos, gu.sph2cart(p_r, theta, phi))

        name = "limb/{}".format(idx)
        limb = xu.body_elem(name, pos)

        theta_j = np.pi + theta
        if theta_j >= 2 * np.pi:
            theta_j = theta_j - 2 * np.pi
        joint_pos = gu.sph2cart(r, theta_j, np.pi - phi)
        for j_idx, axis in enumerate(limb_params["joint_axis"]):
            limb.append(
                xu.joint_elem(
                    "limb{}/{}".format(axis, idx),
                    "hinge",
                    "normal_joint",
                    axis=xu.axis2arr(axis),
                    range_=xu.arr2str(limb_params["joint_range"][j_idx]),
                    pos=xu.arr2str(joint_pos),
                )
            )

        # Create from, to points
        x_f, y_f, z_f = [0.0, 0.0, 0.0]
        x_t, y_t, z_t = gu.sph2cart(r + h, theta, phi)

        # Create actual geom
        # Note as per mujoco docs: The elongated part of the geom connects the
        # two from/to points i.e so we have to handle the r (and p_r) for
        # all the positions.
        limb.append(
            etree.Element(
                "geom",
                {
                    "name": name,
                    "type": "capsule",
                    "fromto": xu.arr2str([x_f, y_f, z_f, x_t, y_t, z_t]),
                    "size": "{}".format(limb_params["limb_radius"]),
                    "density": str(limb_params["limb_density"]),
                },
            )
        )
        x_mid, y_mid, z_mid = gu.sph2cart(r + h / 2, theta, phi)
        limb.append(
            xu.site_elem(
                "limb/mid/{}".format(idx), [x_mid, y_mid, z_mid], site_type
            )
        )

        x_end, y_end, z_end = gu.sph2cart(r + h, theta, phi)
        limb.append(
            xu.site_elem(
                "limb/btm/{}".format(idx), [x_end, y_end, z_end], site_type
            )
        )
        # Site to determine bottom position of geom in global coordinates
        x_end, y_end, z_end = gu.sph2cart(2 * r + h, theta, phi)
        limb.append(
            xu.site_elem(
                "limb/btm_pos/{}".format(idx),
                [x_end, y_end, z_end],
                "btm_pos_site",
            )
        )

        # Site for touch sensor
        limb.append(
            xu.site_elem(
                "limb/touch/{}".format(idx),
                None,
                "touch_site",
                "{}".format(limb_params["limb_radius"] + 0.01),
                xu.arr2str([x_f, y_f, z_f, x_t, y_t, z_t]),
                "capsule",
            )
        )
        return limb, orient

    def set_head_pos(self):
        # Set the head pos to 0, 0, 0 before loading mjsim
        self.unimal.set("pos", xu.arr2str([0.0, 0.0, 0.0]))

        sim = mu.mjsim_from_etree(self.root)
        btm_pos_sites = xu.find_elem(self.unimal, "site", "class", "btm_pos_site")
        btm_pos_site_ids = [
            mu.mj_name2id(sim, "site", site.get("name")) for site in btm_pos_sites
        ]
        sim.step()

        # Get z_axis of all the sites
        z_coords = sim.data.site_xpos[btm_pos_site_ids][:, 2]
        # All the z_axis are < 0. Select the smallest
        btm_most_pos_idx = np.argmin(z_coords)
        head_z = -1 * z_coords[btm_most_pos_idx] + cfg.BODY.FLOOR_OFFSET
        self.unimal.set("pos", xu.arr2str([0, 0, round(head_z, 2)]))

    def _add_actuator(self, body_type, idx, params):
        for axis, gear in zip(params["joint_axis"], params["gear"]):
            name = "{}{}/{}".format(body_type, axis, idx)
            self.actuator.append(xu.actuator_elem(name, gear))

    def _attach(self, site, body_part):
        parent = self.unimal.find(
            './/site[@name="{}"]...'.format(site.get("name"))
        )
        parent.append(body_part)
        return parent

    def _contact_name2id(self, sim, contacts):
        """Converts list of [(name1, name2), ...] to [(id1, id2), ...]."""
        return [
            (mu.mj_name2id(sim, "geom", name1), mu.mj_name2id(sim, "geom", name2))
            for (name1, name2) in contacts
        ]

    def _contact_id2name(self, sim, contacts):
        """Converts list of [(id1, id2), ...] to [(name1, name2), ...]."""
        return [
            (mu.mj_id2name(sim, "geom", id1), mu.mj_id2name(sim, "geom", id2))
            for (id1, id2) in contacts
        ]

    def to_string(self):
        return etree.tostring(self.root, encoding="unicode", pretty_print=True)

    def mutate(self, op=None):
        if not op:
            op = self.choose_mutation()
        else:
            assert op in self.mutation_ops, "Op: {} not supported.".format(op)

        if op == "grow_limb":
            self.grow_limb()
        elif op in ["gear", "dof", "joint_angle"]:
            self.mutate_joint(op)
        elif op == "limb_params":
            self.mutate_limb_params()
        elif op == "density":
            self.mutate_density()
        elif op == "delete_limb":
            self.mutate_delete_limb()

        self.curr_mutation = op

    def choose_mutation(self):
        ops = self.mutation_ops.copy()
        if self.num_limbs == cfg.LIMB.MAX_LIMBS and "grow_limb" in ops:
            ops.remove("grow_limb")
        if self.num_limbs <= 2 and "delete_limb" in ops:
            ops.remove("delete_limb")
        if self.num_limbs == 0:
            ops = ["grow_limb"]
        return random.choice(ops)

    def _sample_child_limb(self):
        limb_list = self.limb_list.copy()
        while True:
            limbs = random.choice(limb_list)
            body = xu.find_elem(
                self.unimal, "body", "name", "limb/{}".format(limbs[0])
            )[0]
            num_children = len(xu.find_elem(body, "body", child_only=True))
            if num_children == 0:
                return limbs
            else:
                limb_list.remove(limbs)

    def mutate_delete_limb(self):
        # Select a child limb(s) to delete
        limb_to_remove = self._sample_child_limb()
        limb_names = set(["limb/{}".format(idx) for idx in limb_to_remove])

        # Remove body from xml
        for limb_idx in limb_to_remove:
            body = xu.find_elem(
                self.unimal, "body", "name", "limb/{}".format(limb_idx)
            )[0]
            body.getparent().remove(body)

        # Remove touch sensor
        sensors = self.root.findall("./sensor")[0]
        for sensor in sensors:
            if sensor.tag != "touch":
                continue
            if xu.name2id(sensor) in limb_to_remove:
                sensors.remove(sensor)

        # Remove exclude contacts
        for exclude in self.contact:
            idx_c1 = int(exclude.get("body1")[-1])
            idx_c2 = int(exclude.get("body2")[-1])
            if idx_c1 in limb_to_remove or idx_c2 in limb_to_remove:
                self.contact.remove(exclude)

        # Remove limbs from limb_metadata
        self.limb_list.remove(limb_to_remove)
        for limb_idx in limb_to_remove:
            del self.limb_metadata[limb_idx]

        # Remove contact pairs
        self.contact_pairs = [
            contact_pair
            for contact_pair in self.contact_pairs
            if not limb_names.intersection(set(contact_pair))
        ]
        self.contact_pairs = set(self.contact_pairs)

        self.num_limbs -= len(limb_to_remove)

        self._align_joints_actuators()

    def mutate_limb_params(self):
        limb_params = self._choose_limb_params()

        limb_to_mutate = random.choice(self.limb_list)
        for limb_idx in limb_to_mutate:
            body = xu.find_elem(
                self.unimal, "body", "name", "limb/{}".format(limb_idx)
            )[0]
            self._mutate_params_of_limb(body, limb_idx, limb_params)

        self.set_head_pos()

    def _mutate_params_of_limb(self, body, limb_idx, limb_params):
        geom = xu.find_elem(body, "geom", "name", "limb/{}".format(limb_idx))[0]

        # Get parent data
        parent_name = self.limb_metadata[limb_idx]["parent_name"]
        p_r = self._get_limb_r(parent_name)

        # New values
        r, h = limb_params["limb_radius"], limb_params["limb_height"]
        _, theta, phi = self.limb_metadata[limb_idx]["orient"]
        x_f, y_f, z_f = [0.0, 0.0, 0.0]
        x_t, y_t, z_t = gu.sph2cart(r + h, theta, phi)
        x_mid, y_mid, z_mid = gu.sph2cart(r + h / 2, theta, phi)
        x_end, y_end, z_end = gu.sph2cart(r + h, theta, phi)
        x_end_btm, y_end_btm, z_end_btm = gu.sph2cart(2 * r + h, theta, phi)
        # New joint pos
        theta_j = np.pi + theta
        if theta_j >= 2 * np.pi:
            theta_j = theta_j - 2 * np.pi
        joint_pos = gu.sph2cart(r, theta_j, np.pi - phi)

        # Update the body pos (Note this is the only place which depends on p_r)
        attach_site = self.limb_metadata[limb_idx]["site"]
        attach_site = xu.find_elem(self.unimal, "site", "name", attach_site)[0]
        pos = xu.str2arr(attach_site.get("pos"))
        pos = xu.add_list(pos, gu.sph2cart(p_r, theta, phi))
        body.set("pos", xu.arr2str(pos, num_decimals=2))

        # Update limb_metadata
        self.limb_metadata[limb_idx]["orient"] = h, theta, phi

        # Update pos of (immediate) child sites.
        sites = xu.find_elem(body, "site", child_only=True)
        for site in sites:
            site_name = site.get("name")
            if "mid" in site_name:
                new_pos = [x_mid, y_mid, z_mid]
            elif "btm" in site_name:
                new_pos = [x_end, y_end, z_end]
            elif "btm_pos" in site_name:
                new_pos = [x_end_btm, y_end_btm, z_end_btm]
            elif "touch" in site_name:
                site.set("fromto", xu.arr2str([x_f, y_f, z_f, x_t, y_t, z_t]))
                site.set("size", str(r + 0.01))
                continue
            else:
                raise ValueError(
                    "site_name {} should not be there!!!".format(site_name)
                )
            site.set("pos", xu.arr2str(new_pos))

        # Update joint pos
        joints = xu.find_elem(body, "joint", child_only=True)
        for joint in joints:
            joint.set("pos", xu.arr2str(joint_pos))

        # Update pos of (immediate) child body elems
        child_bodies = xu.find_elem(body, "body", child_only=True)

        for child_body in child_bodies:
            child_idx = int(child_body.get("name").split("/")[-1])
            # Name of site where it was attached
            site_name = self.limb_metadata[child_idx]["site"]

            # Get child orientation and size
            _, c_theta, c_phi = self.limb_metadata[child_idx]["orient"]

            if "mid" in site_name:
                new_pos = xu.add_list(
                    [x_mid, y_mid, z_mid], gu.sph2cart(r, c_theta, c_phi)
                )
            if "btm" in site_name:
                new_pos = xu.add_list(
                    [x_end, y_end, z_end], gu.sph2cart(r, c_theta, c_phi)
                )

            child_body.set("pos", xu.arr2str(new_pos))

        # Finally, update the actual geom
        geom = xu.find_elem(body, "geom", child_only=True)[0]
        geom.set("fromto", xu.arr2str([x_f, y_f, z_f, x_t, y_t, z_t]))
        geom.set("size", "{}".format(limb_params["limb_radius"]))

    def mutate_density(self):
        self.body_params["limb_density"] = su.sample_from_range(
            cfg.LIMB.DENSITY_RANGE
        )
        self.body_params["torso_density"] = su.sample_from_range(
            cfg.TORSO.DENSITY_RANGE
        )

        limbs = xu.find_elem(self.unimal, "geom", "type", "capsule")
        for limb in limbs:
            # Not needed, safety check
            if "limb" not in limb.get("name"):
                continue
            limb.set("density", str(self.body_params["limb_density"]))

        torsos = xu.find_elem(self.unimal, "geom", "type", "sphere")
        for torso in torsos:
            if "torso" not in torso.get("name"):
                continue
            torso.set("density", str(self.body_params["torso_density"]))

    def mutate_joint(self, op):
        # If true randomly mutate all joints, otherwise only mutate the joints
        # of selected limbs
        mutate_all = random.choice([True, False])

        # Select the corresponding mutation fuction
        if op == "joint_angle":
            op = "angle"
        mutation_func = getattr(self, "_mutate_joint_{}".format(op))

        # Mutate joint
        if mutate_all:
            for limbs in self.limb_list:
                joint_params = self._choose_joint_params(
                    joint_axis=self.limb_metadata[limbs[0]]["joint_axis"]
                )
                mutation_func(limbs, joint_params)
        else:
            limbs = random.choice(self.limb_list)
            joint_params = self._choose_joint_params(
                joint_axis=self.limb_metadata[limbs[0]]["joint_axis"]
            )
            mutation_func(limbs, joint_params)

        self._align_joints_actuators()

    def _mutate_joint_gear(self, limbs, joint_params):
        joint_axis = joint_params["joint_axis"]
        gears = joint_params["gear"]

        for limb_idx in limbs:
            self.limb_metadata[limb_idx]["gear"] = {}
            for axis, gear in zip(joint_axis, gears):
                name = "limb{}/{}".format(axis, limb_idx)
                motor = xu.find_elem(self.actuator, "motor", "name", name)[0]
                motor.set("gear", str(gear))
                self.limb_metadata[limb_idx]["gear"][axis] = gear

    def _mutate_joint_angle(self, limbs, joint_params):
        joint_axis = joint_params["joint_axis"]
        joint_ranges = joint_params["joint_range"]

        for limb_idx in limbs:
            for axis, joint_range in zip(joint_axis, joint_ranges):
                name = "limb{}/{}".format(axis, limb_idx)
                joint = xu.find_elem(self.unimal, "joint", "name", name)[0]
                joint.set("range", xu.arr2str(joint_range))

    def _mutate_joint_dof(self, limbs, joint_params):
        # Resample joint params as the input will have the same joint axis.
        # Little icky but keeps same interface for all joint mutation ops.
        joint_params = self._choose_joint_params()
        joint_axis = joint_params["joint_axis"]

        for limb_idx in limbs:
            # Remove existing joints
            name = "{}/{}".format("limb", limb_idx)
            body = xu.find_elem(self.unimal, "body", "name", name)[0]
            joints = xu.find_elem(body, "joint", child_only=True)
            joint_pos = joints[0].get("pos")
            for joint in joints:
                body.remove(joint)

            # Update limb metadata
            self.limb_metadata[limb_idx]["joint_axis"] = joint_axis
            self.limb_metadata[limb_idx]["gear"] = {}

            # Add new joints
            for insert_pos, axis_ in enumerate(joint_axis):
                body.insert(
                    insert_pos,
                    xu.joint_elem(
                        "limb{}/{}".format(axis_, limb_idx),
                        "hinge",
                        "normal_joint",
                        axis=xu.axis2arr(axis_),
                        range_=xu.arr2str(
                            joint_params["joint_range"][insert_pos]
                        ),
                        pos=joint_pos,
                    ),
                )
                self.limb_metadata[limb_idx]["gear"][axis_] = joint_params[
                    "gear"
                ][insert_pos]

    def grow_limb(self):
        # Try for N = 50 times to grow a limb
        for _ in range(50):
            # Find a site to grow
            choosen_site = self._choose_site()
            if choosen_site is None:
                continue

            choosen_site, limbs2add = choosen_site
            site_type = choosen_site.get("class")

            # Choose limb params
            limb_params = self._choose_limb_params()

            # Get new limbs and sites where to attach them
            limbs, attach_sites, orients = self._get_new_limbs(
                choosen_site, limbs2add, limb_params
            )

            if limbs is None:
                continue

            # Attach the limbs on the site
            parents = []
            exclude_geom_pairs = []

            for cur_limb, cur_site in zip(limbs, attach_sites):
                parents.append(self._attach(cur_site, cur_limb))
                exclude_geom_pairs.append((parents[-1], cur_limb))

            if limbs2add == 2 and site_type == "growth_site":
                exclude_geom_pairs.append((limbs[0], limbs[1]))

            # Check if new unimal is valid
            sim = mu.mjsim_from_etree(self.root)
            sim.step()

            is_symmetric = self._is_symmetric(sim)
            new_contacts = self._new_contacts(sim, exclude_geom_pairs)

            # Update things if new unimal is valid
            if is_symmetric and new_contacts is not None:
                # self._remove_used_sites(parents, attach_sites)
                # Add actuators
                for idx in range(self.limb_idx, self.limb_idx + limbs2add):
                    self._add_actuator("limb", idx, limb_params)
                # Update contacts
                new_contacts = self._contact_id2name(sim, new_contacts)
                self.contact_pairs = self.contact_pairs.union(new_contacts)
                # Update limbs
                self.limb_list.append([xu.name2id(limb_) for limb_ in limbs])
                # Update metadata
                for (l, o, p, a) in zip(limbs, orients, parents, attach_sites):
                    lp = limb_params
                    limb_idx = xu.name2id(l)
                    self.limb_metadata[limb_idx]["joint_axis"] = lp["joint_axis"]
                    self.limb_metadata[limb_idx]["orient"] = o
                    self.limb_metadata[limb_idx]["parent_name"] = p.get("name")
                    self.limb_metadata[limb_idx]["site"] = a.get("name")
                    self.limb_metadata[limb_idx]["gear"] = {}
                    for axis, gear in zip(lp["joint_axis"], lp["gear"]):
                        self.limb_metadata[limb_idx]["gear"][axis] = gear

                # Update mirror dict
                if limbs2add == 2:
                    self._update_mirror_site_dict()

                self.num_limbs += limbs2add
                self.limb_idx += limbs2add
                break

            # Clean up if new unimal not valid
            for cur_parent, cur_limb in zip(parents, limbs):
                cur_parent.remove(cur_limb)

        self.set_head_pos()

    def _get_new_limbs(self, site, limbs2add, limb_params):
        # Types of sites to be made in the new limbs
        if limbs2add == 1:
            new_site_type = "growth_site"
        else:
            new_site_type = "mirror_growth_site"

        # Construct single limb
        limb, orient = self._construct_limb(
            self.limb_idx, site, new_site_type, limb_params
        )

        parent_idx = xu.name2id(site)
        if "torso" not in site.get("name") and self.num_limbs > 0:
            p_orient = self.limb_metadata[parent_idx]["orient"]
            if gu.is_same_orient(orient, p_orient) and "mid" in site.get("name"):
                return None, None, None

        if limbs2add == 1:
            return [limb], [site], [orient]

        # Get mirror site, and mirror orientation
        site_type = site.get("class")
        if site_type == "growth_site":
            mirror_site = site
            r, theta, phi = orient
            # Ensure mirror is not the same
            if theta == 0 or theta == np.pi or phi == np.pi:
                return None, None, None
            mirror_orient = (r, 2 * np.pi - theta, phi)
        else:
            mirror_site_name = self.mirror_sites[site.get("name")]
            mirror_site = xu.find_elem(
                self.unimal, "site", "name", mirror_site_name
            )[0]
            mirror_orient = orient

        # Construct mirror limb
        mirror_limb, _ = self._construct_limb(
            self.limb_idx + 1,
            mirror_site,
            new_site_type,
            limb_params,
            orient=mirror_orient,
        )

        return (
            [limb, mirror_limb],
            [site, mirror_site],
            [orient, mirror_orient],
        )

    def _choose_limb_params(self):
        limb_params = {
            "limb_radius": su.sample_from_range(cfg.LIMB.RADIUS_RANGE),
            "limb_height": su.sample_from_range(cfg.LIMB.HEIGHT_RANGE),
            "gear": su.sample_from_range(cfg.BODY.MOTOR_GEAR_RANGE),
            "limb_density": self.body_params["limb_density"],
        }

        joint_params = self._choose_joint_params()
        limb_params.update(joint_params)
        return limb_params

    def _choose_joint_params(self, joint_axis=None):
        # For each joint axis sample gear and joint range
        if joint_axis is None:
            joint_axis = random.choice(cfg.BODY.JOINT_AXIS)
        joint_range = []
        gear = []
        for axis in joint_axis:
            joint_range.append(
                su.sample_joint_angle_from_list(cfg.BODY.JOINT_ANGLE_LIST)
            )
            gear.append(su.sample_from_range(cfg.BODY.MOTOR_GEAR_RANGE))

        return {
            "joint_axis": joint_axis,
            "joint_range": joint_range,
            "gear": gear,
        }

    def _choose_torso_params(self):
        return {
            "torso_radius": su.sample_from_range(cfg.TORSO.HEAD_RADIUS_RANGE),
            "torso_density": self.body_params["torso_density"],
        }

    def _remove_used_sites(self, parents, attachment_sites):
        # Remove sites as they are no longer free
        for cur_parent, cur_site in zip(parents, attachment_sites):
            if "torso" in cur_site.get("name"):
                continue
            # For the case when limb2add = 2 and growth site. List
            # would have the same site twice.
            try:
                cur_parent.remove(cur_site)
            except ValueError:
                continue

    def _choose_site(self):
        """Choose a free site, and how many limbs to add at the site."""

        # Choose a torso first, and then choose within the torso child elems

        torso_idx = random.choice(self.growth_torso)
        torso = xu.find_elem(
            self.root, "body", "name", "torso/{}".format(torso_idx)
        )[0]
        free_sites = xu.find_elem(torso, "site", "class", "growth_site")
        free_sites.extend(
            xu.find_elem(torso, "site", "class", "mirror_growth_site")
        )
        site = random.choice(free_sites)

        if site.get("class") == "growth_site":
            limbs2add = random.randint(1, 2)
        else:
            # For mirror_growth sites always return the lower name number.
            # For e.g if we select a site with name "limb/mid/3" and the mirror
            # site is "limb/mid/1". Then we return the mirror site. This
            # prevents duplicates.
            mirror_site_name = self.mirror_sites[site.get("name")]
            if (
                mirror_site_name < site.get("name")
                and "torso" not in mirror_site_name
            ):
                site = xu.find_elem(
                    self.unimal, "site", "name", mirror_site_name
                )[0]
            limbs2add = 2

        if limbs2add + self.num_limbs > cfg.LIMB.MAX_LIMBS:
            return None

        return site, limbs2add

    def _is_symmetric(self, sim):
        """Check if current unimal is symmetric along BODY.SYMMETRY_PLANE."""

        # Get unimal center of mass (com)
        head_idx = sim.model.body_name2id(HEAD)
        unimal_com = sim.data.subtree_com[head_idx, :]
        # Center of mass should have zero component along axis normal to
        # cfg.BODY.SYMMETRY_PLANE.
        normal_axis = cfg.BODY.SYMMETRY_PLANE.index(0)
        return unimal_com[normal_axis] == 0

    def _new_contacts(self, sim, exclude_geom_pairs):
        """New contacts should only be between exclude_geom_pairs."""

        # exclude_geom_pairs will contain all parent child geom pairs being
        # added and if two geoms are added to the same parent, it will
        # contain that pair also. There should be no new contact other than
        # these pairs and previous such relations as it would indicate self-
        # intersection.

        num_contacts = sim.data.ncon
        contacts = sim.data.contact[:num_contacts]

        exclude_geom_pairs = [
            tuple(
                sorted(
                    (
                        mu.mj_name2id(sim, "geom", geom1.get("name")),
                        mu.mj_name2id(sim, "geom", geom2.get("name")),
                    )
                )
            )
            for (geom1, geom2) in exclude_geom_pairs
        ]

        prev_contact_pairs = self._contact_name2id(sim, self.contact_pairs)
        prev_contact_pairs = [tuple(sorted(_)) for _ in prev_contact_pairs]

        for contact in contacts:
            cur_pair = tuple(sorted((contact.geom1, contact.geom2)))
            if (
                cur_pair not in prev_contact_pairs
                and cur_pair not in exclude_geom_pairs
            ):
                return None
                # print(
                #     mu.mj_id2name(sim, "geom", contact.geom1),
                #     mu.mj_id2name(sim, "geom", contact.geom2)
                # )

        return exclude_geom_pairs

    def _update_joint_axis(self):
        sim = mu.mjsim_from_etree(self.root)
        sim.step()
        limbs = xu.find_elem(self.root, "body")
        limbs = [elem for elem in limbs if "torso" not in elem.get("name")]
        for limb in limbs:
            name = limb.get("name")
            # z-axis of geom frame always points in the direction end -> start
            # (or to --> from). Valid rotations are about x and y axis of the
            # geom frame.
            geom_frame = sim.data.get_geom_xmat(name).reshape(3, 3)
            joints = xu.find_elem(limb, "joint", child_only=True)
            for joint in joints:
                joint_name = joint.get("name")
                if joint_name[4] == "x":
                    curr_axis = geom_frame[:, 0]
                elif joint_name[4] == "y":
                    curr_axis = geom_frame[:, 1]
                joint.set("axis", xu.arr2str(curr_axis, num_decimals=4))

    def _update_mirror_site_dict(self):
        site_name = "limb/mid/{}".format(self.limb_idx)
        mirror_name = "limb/mid/{}".format(self.limb_idx + 1)
        self.mirror_sites[site_name] = mirror_name
        self.mirror_sites[mirror_name] = site_name

        site_name = "limb/btm/{}".format(self.limb_idx)
        mirror_name = "limb/btm/{}".format(self.limb_idx + 1)
        self.mirror_sites[site_name] = mirror_name
        self.mirror_sites[mirror_name] = site_name

    def _align_joints_actuators(self):
        """Ensure that the joint order in body and actuators is aligned."""
        # Delete all gears
        motors = xu.find_elem(self.actuator, "motor")
        for motor in motors:
            self.actuator.remove(motor)

        # Add gears corresponding to joints
        joints = xu.find_elem(self.unimal, "joint", "class", "normal_joint")
        for joint in joints:
            name = joint.get("name")
            axis = name.split("/")[0][-1]
            gear = self.limb_metadata[xu.name2id(joint)]["gear"][axis]
            self.actuator.append(xu.actuator_elem(name, gear))

    def _add_sensors(self):
        sensor = self.root.findall("./sensor")[0]
        for s in sensor:
            sensor.remove(s)

        # Add imu sensors
        sensor.append(xu.sensor_elem("accelerometer", "torso_accel", "root"))
        sensor.append(xu.sensor_elem("gyro", "torso_gyro", "root"))
        # Add torso velocity sensor
        sensor.append(xu.sensor_elem("velocimeter", "torso_vel", "root"))
        # Add subtreeangmom sensor
        sensor.append(
            etree.Element("subtreeangmom", {
                "name": "unimal_am", "body": "torso/0"
            })
        )
        # Add touch sensors
        bodies = xu.find_elem(self.root, "body")
        for body in bodies:
            body_name = body.get("name").split("/")
            site_name = "{}/touch/{}".format(body_name[0], body_name[1])
            sensor.append(xu.sensor_elem("touch", body.get("name"), site_name))

    def _remove_permanent_contacts(self):
        contact_pairs = xu.find_elem(self.contact, "exclude")
        for cp in contact_pairs:
            self.contact.remove(cp)

    def _exclude_permanent_contacts(self):
        # Enable filterparent
        flag = xu.find_elem(self.root, "flag")[0]
        flag.set("filterparent", str("enable"))

        sim = mu.mjsim_from_etree(self.root)
        sim.forward()
        contact_pairs = mu.get_active_contacts(sim)
        for geom1, geom2 in contact_pairs:
            self.contact.append(xu.exclude_elem(geom1, geom2))

        flag.set("filterparent", str("disable"))

    def _add_floor(self):
        self.worldbody.insert(
            1,
            etree.fromstring(
                '<geom name="floor" type="plane" pos="0 0 0" size="50 50 1" material="grid"/>'
            ),
        )

    def save_image(self):
        sim = mu.mjsim_from_etree(self.root)
        sim.step()
        frame = sim.render(
            cfg.IMAGE.WIDTH,
            cfg.IMAGE.HEIGHT,
            depth=False,
            camera_name=cfg.IMAGE.CAMERA,
            mode="offscreen",
        )
        # Rendered images are upside down
        frame = frame[::-1, :, :]
        imageio.imwrite(fu.id2path(self.id, "images"), frame)

    def _get_limb_r(self, name):
        elem = xu.find_elem(self.unimal, "geom", "name", name)[0]
        return float(elem.get("size"))

    def _before_save(self):
        self._align_joints_actuators()
        self._add_sensors()
        self._update_joint_axis()
        self._exclude_permanent_contacts()

    def save(self):
        self._before_save()
        xml_path = os.path.join(cfg.OUT_DIR, "xml", "{}.xml".format(self.id))
        xu.save_etree_as_xml(self.tree, xml_path)
        if self.parent_id:
            mutation_op = self.curr_mutation
        else:
            mutation_op = ""
        init_state = {
            "xml_path": xml_path,
            "contact_pairs": self.contact_pairs,
            "num_limbs": self.num_limbs,
            "limb_idx": self.limb_idx,
            "num_torso": self.num_torso,
            "torso_list": self.torso_list,
            "body_params": self.body_params,
            "limb_list": self.limb_list,
            "limb_metadata": self.limb_metadata,
            "mirror_sites": self.mirror_sites,
            "dof": len(xu.find_elem(self.actuator, "motor")),
            "parent_id": self.parent_id,
            "mutation_op": mutation_op,
            "growth_torso": self.growth_torso,
        }
        save_path = os.path.join(
            cfg.OUT_DIR, "unimal_init", "{}.pkl".format(self.id)
        )
        fu.save_pickle(init_state, save_path)
