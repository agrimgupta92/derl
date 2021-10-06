import math
from copy import deepcopy

import numpy as np
from scipy import signal

from derl.config import cfg
from derl.utils import sample as su
from derl.utils import xml as xu

# Since all terrain elems are made from box and plane len/size etc will always
# be half len which matches with mujoco specs.
WALL_LENGTH = 0.5


class Terrain:
    """Module for procedural generation of terrain."""

    def __init__(self, random_state=None):
        self.terrain_types = cfg.TERRAIN.TYPES
        self.np_random = random_state

        self.width = cfg.TERRAIN.SIZE[1]
        self.divs = cfg.HFIELD.NUM_DIVS
        self.segms = []

        if cfg.HFIELD.DIM == 1:
            self.hfield = np.zeros((1, cfg.TERRAIN.SIZE[0] * 2 * self.divs))
        else:
            self.hfield = np.zeros(
                (self.width * 2 * self.divs, cfg.TERRAIN.SIZE[0] * 2 * self.divs)
            )

        self.asset_elem = []  # stores the xml elem
        self.asset_hfield = []  # stores the actual hfield values

        self.assert_cfg()

    def _create_flat(self, size=None):
        """Create flat terrain."""
        prev_segm = deepcopy(self.segms[-1])

        if size:
            flat_len = size
        else:
            flat_len = su.sample_from_range(
                cfg.TERRAIN.FLAT_LENGTH_RANGE, rng_state=self.np_random
            )

        # Trucate if flat_len addition exceeds terrain size
        curr_end = (prev_segm["end_x"] + flat_len * 2) * self.divs
        if curr_end > self.max_lim:
            flat_len = (self.max_lim / self.divs - prev_segm["end_x"]) / 2
            if not flat_len:
                return []

        pos = prev_segm["pos"]
        pos[0] += prev_segm["len"] + flat_len
        name = "floor/{}".format(prev_segm["idx"] + 1)
        size = [flat_len, self.width, prev_segm["h"]]
        terr_elem = xu.floor_segm(name, pos, size, "box")

        prev_end = int(prev_segm["end_x"] * self.divs)
        curr_end = int((prev_segm["end_x"] + flat_len * 2) * self.divs)

        self.hfield[:, prev_end:curr_end] = prev_segm["h"]

        self.segms.append(
            {
                "pos": pos,
                "len": flat_len,
                "h": prev_segm["h"],
                "idx": prev_segm["idx"] + 1,
                "end_x": round(prev_segm["end_x"] + flat_len * 2, 2),
                "segm_type": "flat",
            }
        )

        return [terr_elem]

    def _create_curve_slope(self, nrow, ncol):
        """Create hfield data for a curved slope."""
        curve_h = su.sample_from_range(
            cfg.TERRAIN.CURVE_HEIGHT_RANGE, rng_state=self.np_random
        )
        hfield_data = np.zeros((nrow, ncol))
        curve = np.linspace(0, np.pi, num=ncol)
        curve = np.sin(curve)
        curve *= curve_h
        hfield_data[:, :] = curve
        return hfield_data

    def _create_rugged_square(self, nrow, ncol):
        """Create a bumpy terrain by clipping a triangle wave."""
        hfield_data = np.zeros((nrow, ncol))
        idx = 0
        while True:
            width = su.sample_from_list(
                range(10, 20, 1), rng_state=self.np_random
            )
            f = su.sample_from_list([10, 15], rng_state=self.np_random)
            y = np.linspace(0, f * np.pi, num=ncol)
            y = signal.sawtooth(y, width=0.5) + 1
            c = su.sample_from_range(
                cfg.TERRAIN.RUGGED_SQUARE_CLIP_RANGE, rng_state=self.np_random
            )
            y = np.clip(y, None, c)
            y = y.reshape((1, -1))
            hfield_data[idx : idx + width, :] = y
            idx += width
            if idx >= nrow:
                break

        return hfield_data

    def _create_steps(self, nrow, ncol):
        """Create steps which go up and then down."""
        num_steps = int(cfg.TERRAIN.NUM_STEPS / 2)
        repeat_unit = ncol / (num_steps * 2)
        hfield_data = np.zeros((nrow, ncol))
        heights = list(range(0, num_steps, 1)) + list(
            range(num_steps - 2, -1, -1)
        )
        heights = [round(_ * cfg.TERRAIN.STEP_HEIGHT, 2) for _ in heights]
        repeats = [repeat_unit] * ((num_steps * 2) - 1)
        repeats[num_steps - 1] *= 2
        hfield_data[:, :] = np.repeat(heights, repeats)
        return hfield_data

    def _create_hfield(self, segm_type):
        """Create hfield obstacles."""
        prev_segm = deepcopy(self.segms[-1])
        # Sample len and h of hfield
        if segm_type == "steps":
            len_range = cfg.TERRAIN.STEP_LENGTH_RANGE
        else:
            len_range = cfg.TERRAIN.HFIELD_LENGTH_RANGE

        hfield_len = su.sample_from_range(len_range, rng_state=self.np_random)
        # Check hfield bounds
        prev_end = int(prev_segm["end_x"] * self.divs)
        curr_end = int((prev_segm["end_x"] + hfield_len * 2) * self.divs)
        if curr_end > self.max_lim:
            return []

        # Create hfield asset
        pos = prev_segm["pos"]
        pos[0] += prev_segm["len"] + hfield_len
        name = "floor/{}".format(prev_segm["idx"] + 1)
        nrow = self.width * self.divs * 2
        ncol = curr_end - prev_end

        # Create hfield data
        hfield_elem = xu.floor_segm(name, pos, None, "hfield", "hfield")
        if segm_type == "curve_slope":
            hfield_data = self._create_curve_slope(nrow, ncol)
        elif segm_type == "rugged_square":
            hfield_data = self._create_rugged_square(nrow, ncol)
        elif segm_type == "steps":
            hfield_data = self._create_steps(nrow, ncol)

        # Offset hfield by prev_segm height
        hfield_data += prev_segm["h"]
        if cfg.HFIELD.DIM == 1:
            self.hfield[:, prev_end:curr_end] = hfield_data[0, :]
        else:
            self.hfield[:, prev_end:curr_end] = hfield_data

        # hfield data in mujoco sim.data should be between [0, 1]
        max_z = np.max(hfield_data)
        hfield_data = hfield_data / max_z
        self.asset_hfield.append(hfield_data.reshape(-1))

        # finally, create the asset elem for hfield
        size = [hfield_len, self.width, max_z, 0.1]
        hfield_asset = xu.hfield_asset(name, nrow, ncol, size)
        self.asset_elem.append(hfield_asset)

        self.segms.append(
            {
                "pos": pos,
                "len": hfield_len,
                "h": prev_segm["h"],
                "idx": prev_segm["idx"] + 1,
                "end_x": round(prev_segm["end_x"] + hfield_len * 2, 2),
                "segm_type": segm_type,
            }
        )

        return [hfield_elem]

    def create_boundary_walls(self):
        l, w, h = cfg.TERRAIN.SIZE
        center = l - cfg.TERRAIN.START_FLAT
        boundary = []
        wall_height = 2
        # Front, back, left and right are when viewing in +x direction

        # Add front wall
        pos = [center + l + WALL_LENGTH, 0, wall_height - h]
        size = [WALL_LENGTH, w + 2 * WALL_LENGTH, wall_height]
        boundary.append(
            xu.floor_segm("boundary/front", pos, size, "box", "boundary")
        )
        # Add back wall
        pos = [-(cfg.TERRAIN.START_FLAT + WALL_LENGTH), 0, wall_height - h]
        boundary.append(
            xu.floor_segm("boundary/back", pos, size, "box", "boundary")
        )
        # Add right wall
        pos = [center, w + WALL_LENGTH, wall_height - h]
        size = [l, WALL_LENGTH, wall_height]
        boundary.append(
            xu.floor_segm("boundary/right", pos, size, "box", "boundary")
        )

        # Add left wall
        pos = [center, -(w + WALL_LENGTH), wall_height - h]
        boundary.append(
            xu.floor_segm("boundary/left", pos, size, "box", "boundary")
        )
        # Modify hfield
        self.add_padding(int(WALL_LENGTH * 2 * self.divs), wall_height * 2)
        return boundary

    def _create_terrain_segm(self, segm_type, size=None):
        hfield_segms = ["curve_slope", "rugged_square", "steps"]
        if segm_type == "flat":
            return self._create_flat(size=size)
        elif segm_type in hfield_segms:
            return self._create_hfield(segm_type)
        else:
            raise ValueError("segm_type: {} not supported.".format(segm_type))

    def _create_initial_flat_segm(self):
        # Add the first piece of flat terrain
        start_flat_len = cfg.TERRAIN.START_FLAT
        size = [start_flat_len] + cfg.TERRAIN.SIZE[1:]
        name = "floor/0"
        pos = [0, 0, 0]
        terr_elem = xu.floor_segm(
            name, pos, size, "box", incline=cfg.TERRAIN.INCLINE_ANGLE
        )
        self.segms.append(
            {
                "pos": [0, 0, 0],
                "len": start_flat_len,
                "h": cfg.TERRAIN.SIZE[2],
                "idx": 0,
                "end_x": round(start_flat_len * 2, 2),
                "segm_type": "flat",
            }
        )
        end_idx = int(start_flat_len * 2 * self.divs)
        if cfg.ENV.TASK in ["incline", "push_box_incline"]:
            angle = np.deg2rad(abs(cfg.TERRAIN.INCLINE_ANGLE))
            origin_h = np.tan(angle) * cfg.TERRAIN.SIZE[0]
            if cfg.TERRAIN.INCLINE_ANGLE < 0:
                base_list = np.asarray(range(0, end_idx))
            else:
                base_list = np.asarray(range(end_idx - 1, -1, -1))
            self.hfield[:, :end_idx] = (
                (np.tan(angle) * base_list) / self.divs + 1 - origin_h
            )
        else:
            self.hfield[:, :end_idx] = cfg.TERRAIN.SIZE[2]
        return terr_elem

    def create_corridor(self):
        xml_elems = []
        # Add initial flat segm
        xml_elems.append(self._create_initial_flat_segm())

        self.max_lim = self.hfield.shape[1]
        # Each iteration of the loop will add one flat segm, and one randomly
        # sampled obstacle from the list cfg.TERRAIN.TYPES
        while self.segms[-1]["end_x"] < cfg.TERRAIN.SIZE[0] * 2:
            trr_type = su.sample_from_list(self.terrain_types, self.np_random)
            flat = self._create_terrain_segm("flat")
            obstacle = self._create_terrain_segm(trr_type)
            # Check if obstacle was succesfully made
            if obstacle:
                terr_elems = flat + obstacle
            else:
                terr_elems = flat
            for terr_elem in terr_elems:
                xml_elems.append(terr_elem)

        return xml_elems

    def create_arena(self):
        xml_elems = []
        self.segms.append(
            {
                "pos": [0, 0, 0],
                "len": 0,
                "h": cfg.TERRAIN.SIZE[2],
                "idx": -1,
                "end_x": 0,
                "segm_type": "flat",
            }
        )

        # Each iteration of the loop will add one flat segm, and one randomly
        # sampled obstacle from the list cfg.TERRAIN.TYPES
        self.max_lim = (
            self.hfield.shape[1] / 2 - cfg.TERRAIN.CENTER_FLAT * self.divs
        )
        while self.segms[-1]["end_x"] < cfg.TERRAIN.SIZE[0] * 2:
            trr_type = su.sample_from_list(self.terrain_types, self.np_random)
            obstacle = self._create_terrain_segm(trr_type)
            flat = self._create_terrain_segm("flat")
            # Check if obstacle was succesfully made
            if obstacle:
                trr_elems = flat + obstacle
            else:
                trr_elems = flat
            for trr_elem in trr_elems:
                xml_elems.append(trr_elem)

            if (
                self.segms[-1]["end_x"]
                == cfg.TERRAIN.SIZE[0] - cfg.TERRAIN.CENTER_FLAT
            ):
                self.max_lim = self.hfield.shape[1]
                xml_elems.append(
                    self._create_terrain_segm(
                        "flat", size=cfg.TERRAIN.CENTER_FLAT
                    )[0]
                )

        return xml_elems

    def create_scene(self):
        if cfg.ENV.TASK in ["locomotion", "obstacle", "goal_follow", "incline", "manipulation", "push_box_incline"]:
            xml_elems = self.create_corridor()
        else:
            xml_elems = self.create_arena()

        if cfg.TERRAIN.BOUNDARY_WALLS:
            xml_elems.extend(self.create_boundary_walls())
        # Pad hfield with cfg.HFIELD.GAP_DEPTH to handle hfield obs when
        # agent is at edges
        padding = cfg.HFIELD.PADDING * self.divs
        self.add_padding(padding, cfg.HFIELD.GAP_DEPTH)
        self.hfield = np.round_(self.hfield, 3)

        return xml_elems

    def add_padding(self, padding, pad_value):
        if cfg.HFIELD.DIM == 1:
            pad_width = [(0, 0), (padding, padding)]
        else:
            pad_width = padding

        self.hfield = np.pad(
            self.hfield, pad_width, mode="constant", constant_values=pad_value,
        )

    def exclude_floor_contacts(self, root):
        contact = root.findall("./contact")[0]
        num_floors = self.segms[-1]["idx"]
        # Add pair of adjacent floor bodies between which we will exclude
        # contacts
        for (idx1, idx2) in zip(range(num_floors), range(1, num_floors)):
            contact.append(
                xu.exclude_elem("floor/{}".format(idx1), "floor/{}".format(idx2))
            )

    def modify_xml_step(self, env, root, tree):
        worldbody = root.findall("./worldbody")[0]

        insert_pos = 1
        xml_elems = self.create_scene()
        for elem in xml_elems:
            worldbody.insert(insert_pos, elem)
            insert_pos += 1

        # Add all hfield asset elems
        asset = root.findall("./asset")[0]
        for elem in self.asset_elem:
            asset.append(elem)

        env.metadata["hfield"] = self.hfield
        # np.save("outfile.npy", self.hfield)
        # xu.save_etree_as_xml(tree, "1.xml")

    def modify_sim_step(self, env, sim):
        start_pos = 0
        for idx, hfield in enumerate(self.asset_hfield):
            sim.model.hfield_data[start_pos : start_pos + len(hfield)] = hfield
            start_pos += len(hfield)

    def observation_step(self, env, sim):
        x_pos, y_pos, _ = sim.data.get_body_xpos("torso/0")
        row_idx, col_idx = self.pos_to_idx([x_pos, y_pos])
        return {"hfield_idx": np.asarray([row_idx, col_idx])}

    def pos_to_idx(self, pos):
        x_pos, y_pos = pos
        start_offset = cfg.TERRAIN.START_FLAT + cfg.HFIELD.PADDING
        if cfg.TERRAIN.BOUNDARY_WALLS:
            start_offset += WALL_LENGTH * 2
        start_offset *= self.divs
        # Get agent idx in hfield, row --> y, col --> x
        col_idx = int(start_offset + math.floor(x_pos * cfg.HFIELD.NUM_DIVS))
        row_idx = 0
        if cfg.HFIELD.DIM == 2:
            row_idx = int(
                self.hfield.shape[0] / 2 - math.floor(y_pos * cfg.HFIELD.NUM_DIVS)
            )
        return [row_idx, col_idx]

    def idx_to_pos(self, idx):
        row_idx, col_idx = idx
        y_pos = (self.hfield.shape[0] / 2 - row_idx) / cfg.HFIELD.NUM_DIVS

        start_offset = cfg.TERRAIN.START_FLAT + cfg.HFIELD.PADDING
        if cfg.TERRAIN.BOUNDARY_WALLS:
            start_offset += WALL_LENGTH * 2
        start_offset *= cfg.HFIELD.NUM_DIVS
        x_pos = (col_idx - start_offset) / cfg.HFIELD.NUM_DIVS
        return x_pos, y_pos

    def assert_cfg(self):
        # We make quite a few assumptions in the code. Check here if cfg
        # params match the assumptions.

        # For steps repeat_unit should be integer (refer _create_steps)
        if "steps" in cfg.TERRAIN.TYPES:
            list_ = set(np.arange(*cfg.TERRAIN.STEP_LENGTH_RANGE))
            list_.add(cfg.TERRAIN.STEP_LENGTH_RANGE[1])
            list_ = [l * cfg.HFIELD.NUM_DIVS * 2 for l in list_]
            assert all([l % cfg.TERRAIN.NUM_STEPS == 0 for l in list_])

        # Planar beings can't avoid!
        if cfg.HFIELD.DIM == 1:
            not_planar = set(["rugged_square"])
            assert len(not_planar.intersection(set(cfg.TERRAIN.TYPES))) == 0

        if cfg.ENV.TASK not in [
            "locomotion",
            "obstacle",
            "incline",
            "manipulation",
            "push_box_incline"
        ]:
            assert cfg.TERRAIN.START_FLAT == 0
            assert cfg.TERRAIN.CENTER_FLAT != 0

    def _check_all_int(self, list_):
        return all(isinstance(x, int) for x in list_)
