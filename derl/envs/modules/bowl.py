import math

import numpy as np
from scipy import ndimage

from derl.config import cfg
from derl.utils import xml as xu

_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).


class Bowl:
    """Module for procedural generation of bowl shaped terrain."""

    def __init__(self, random_state=None):
        self.np_random = random_state

        self.side = cfg.TERRAIN.SIZE[0]
        self.divs = cfg.HFIELD.NUM_DIVS

        self.hfield = np.zeros(
            (self.side * 2 * self.divs, self.side * 2 * self.divs)
        )

        self.asset_hfield = None

    def create_bowl(self):
        name = "foor/0"
        hfield_elem = xu.floor_segm(name, [0, 0, 0], None, "hfield", "hfield")

        # Sinusoidal bowl shape
        res = self.side * 2 * self.divs
        row_grid, col_grid = np.ogrid[-1 : 1 : res * 1j, -1 : 1 : res * 1j]
        radius = np.clip(np.sqrt(col_grid ** 2 + row_grid ** 2), 0.04, 1)
        bowl_shape = 0.5 - np.cos(2 * np.pi * radius) / 2

        # Random smooth bumps
        terrain_size = self.side * 2
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = self.np_random.uniform(
            _TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res)
        )
        smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))

        # Terrain is elementwise product
        max_z = cfg.TERRAIN.BOWL_MAX_Z
        self.hfield = smooth_bumps * bowl_shape * max_z
        self.asset_hfield = smooth_bumps * bowl_shape
        # Finally, create the asset elem for hfield
        size = [self.side, self.side, max_z, 0.1]
        hfield_asset = xu.hfield_asset(name, res, res, size)

        return [hfield_elem], [hfield_asset]

    def modify_xml_step(self, env, root, tree):
        worldbody = root.findall("./worldbody")[0]

        insert_pos = 1
        xml_elems, hfield_assets = self.create_bowl()
        for elem in xml_elems:
            worldbody.insert(insert_pos, elem)
            insert_pos += 1

        # Add all hfield asset elems
        asset = root.findall("./asset")[0]
        for elem in hfield_assets:
            asset.append(elem)

        env.metadata["hfield"] = self.hfield

    def modify_sim_step(self, env, sim):
        sim.model.hfield_data[0 : self.asset_hfield.size] = self.asset_hfield.ravel()

    def observation_step(self, env, sim):
        x_pos, y_pos, _ = sim.data.get_body_xpos("torso/0")
        row_idx, col_idx = self.pos_to_idx([x_pos, y_pos])
        return {"hfield_idx": np.asarray([row_idx, col_idx])}

    def pos_to_idx(self, pos):
        x_pos, y_pos = pos
        # Get agent idx in hfield, row --> y, col --> x
        start_offset = self.side * self.divs
        col_idx = int(start_offset + math.floor(x_pos * cfg.HFIELD.NUM_DIVS))
        row_idx = int(
            self.hfield.shape[0] / 2 - math.floor(y_pos * cfg.HFIELD.NUM_DIVS)
        )
        return [row_idx, col_idx]

    def idx_to_pos(self, idx):
        row_idx, col_idx = idx
        start_offset = self.side * self.divs
        y_pos = (self.hfield.shape[0] / 2 - row_idx) / cfg.HFIELD.NUM_DIVS
        x_pos = (col_idx - start_offset) / cfg.HFIELD.NUM_DIVS
        return x_pos, y_pos
