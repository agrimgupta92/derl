import numpy as np

from derl.config import cfg


class PatrolGoals:
    """Generate goals for patrol task."""

    def __init__(self, random_state=None):

        self.np_random = random_state
        self.goal_pos = None
        self.cur_goal = "backward"

        # Sites which should be visible when rendering
        self.markers = []

    def toggle_goal(self, env):
        gs = 0.5
        gh = 0.01

        l, w, h = cfg.TERRAIN.SIZE

        if self.cur_goal == "backward":
            self.goal_pos = [cfg.TERRAIN.PATROL_HALF_LEN, 0, h + 0.1]
            self.cur_goal = "forward"
        else:
            self.goal_pos = [-cfg.TERRAIN.PATROL_HALF_LEN, 0, h + 0.1]
            self.cur_goal = "backward"

        self.markers = []
        self.markers.append(
            {
                "label": "",
                "size": np.array([gs, gs, gh]),
                "rgba": np.array([1, 0, 0, 0.4]),
                "pos": np.array(self.goal_pos),
            }
        )
        env.metadata["markers"] = self.markers

    def modify_xml_step(self, env, root, tree):
        self.toggle_goal(env)

    def modify_sim_step(self, env, sim):
        pass

    def observation_step(self, env, sim):
        # Convert box pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")

        goal_rel_pos = self.goal_pos - torso_pos
        goal_state = goal_rel_pos.dot(torso_frame).ravel()
        return {"goal": goal_state}
