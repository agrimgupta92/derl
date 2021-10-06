from derl.config import cfg
from derl.utils import xml as xu


class Floor:
    """Module to add floor to env."""

    def __init__(self, random_state=None):
        self.np_random = random_state

    def modify_xml_step(self, env, root, tree):
        size = cfg.TERRAIN.SIZE
        floor_elem = xu.floor_segm("floor/0", [0, 0, 0], size, "box", "grid")
        worldbody = root.findall("./worldbody")[0]
        worldbody.insert(1, floor_elem)

    def modify_sim_step(self, env, sim):
        return

    def observation_step(self, env, sim):
        return {}
