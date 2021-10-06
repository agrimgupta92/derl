"""Configuration file (powered by YACS)."""

import copy
import os

from derl.yacs import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ----------------------------------------------------------------------------#
# XML template params
# ----------------------------------------------------------------------------#
# Refer mujoco docs for what each param does

_C.XML = CN()

_C.XML.NJMAX = 1000

_C.XML.NCONMAX = 200

_C.XML.GEOM_CONDIM = 3

_C.XML.GEOM_FRICTION = [0.7, 0.1, 0.1]

_C.XML.FILTER_PARENT = "enable"

_C.XML.SHADOWCLIP = 0.5

# ----------------------------------------------------------------------------#
# Body Options
# ----------------------------------------------------------------------------#
_C.BODY = CN()

# Vertical distance between the bottom most point of unimal and floor
_C.BODY.FLOOR_OFFSET = 0.2

# Plane of bilateral symmetry. e.g. (1, 0, 1): xz plane of symmetry
_C.BODY.SYMMETRY_PLANE = (1, 0, 1)

_C.BODY.JOINT_ANGLE_LIST = [
    [-30, 0],
    [0, 30],
    [-30, 30],
    [-45, 45],
    [-45, 0],
    [0, 45],
    [-60, 0],
    [0, 60],
    [-60, 60],
    [-90, 0],
    [0, 90],
    [-60, 30],
    [-30, 60],
]

_C.BODY.MOTOR_GEAR_RANGE = [150, 300, 50]

# Joint axis in the geom frame. x means we will choose the x-axis in geom frame
# as the joint axis.
_C.BODY.JOINT_AXIS = ["x", "y", "xy"]

# ----------------------------------------------------------------------------#
# Torso Options
# ----------------------------------------------------------------------------#
_C.TORSO = CN()

# Body elems are always sphere.
_C.TORSO.HEAD_RADIUS_RANGE = [0.10]

# Num of torso elems, typically 1 head (sphere) and 1-3 capsules
_C.TORSO.NUM_TORSO_RANGE = [0]

_C.TORSO.RADIUS_RANGE = [0.2]

_C.TORSO.HEIGHT_RANGE = [0.15]

# Condim (see mujoco docs) for torso
_C.TORSO.CONDIM = 3

# Density of the torso
_C.TORSO.DENSITY_RANGE = [500, 1000, 100]

# ----------------------------------------------------------------------------#
# Limb Options
# ----------------------------------------------------------------------------#
_C.LIMB = CN()

# Range is interpreted as [start, stop, step]. With start and stop included.
# if range is just one element then it's considered as fixed.
_C.LIMB.RADIUS_RANGE = [0.05]

_C.LIMB.HEIGHT_RANGE = [0.2, 0.4, 0.1]

# Density of the limbs
_C.LIMB.DENSITY_RANGE = [500, 1000, 100]

# Number of limbs in the unimal. Limbs does not include the head. Range is
# inclusive.
_C.LIMB.NUM_LIMBS_RANGE = [2, 8, 1]

# Max number of allowed limbs. When this limit is reached mutation won't
# grow limbs
_C.LIMB.MAX_LIMBS = 10

# ----------------------------------------------------------------------------#
# Unimal Env Options
# ----------------------------------------------------------------------------#
_C.ENV = CN()

_C.ENV.FORWARD_REWARD_WEIGHT = 1.0

_C.ENV.AVOID_REWARD_WEIGHT = 100.0

_C.ENV.CTRL_COST_WEIGHT = 1e-3

_C.ENV.HEALTHY_REWARD = 0.0

_C.ENV.STAND_REWARD_WEIGHT = 0.0

# See ReachHabitatReward for details
_C.ENV.REACH_REWARD_PARAMS = [100.0, 0.0, 10]

_C.ENV.STATE_RANGE = (-100.0, 100.0)

_C.ENV.Z_RANGE = (-0.1, float("inf"))

_C.ENV.RESET_NOISE_SCALE = 5e-3

# Healthy reward is 1 if head_pos >= STAND_HEIGHT_RATIO * head_pos in
# the xml i.e the original height of the unimal.
_C.ENV.STAND_HEIGHT_RATIO = 0.5

# List of modules to add to the env. Modules will be added in the same order
_C.ENV.MODULES = ["Floor", "Agent"]

# Agent name if you are not using unimal but want to still use the unimal env
_C.ENV.WALKER = ""

# Types of proprioceptive obs to include
_C.ENV.OBS_TYPES = ["position", "velocity", "imu_vel", "touch", "extremities"]

# Keys to keep in SelectKeysWrapper
_C.ENV.KEYS_TO_KEEP = ["proprioceptive"]

# Skip position of free joint (or x root joint) in position observation for
# translation invariance
_C.ENV.SKIP_SELF_POS = True

# Specify task. Can be locomotion, manipulation
_C.ENV.TASK = "locomotion"

# Optional wrappers to add to task. Most wrappers for a task will eventually be
# hardcoded in make_env_task func. Put wrappers which you want to experiment
# with.
_C.ENV.WRAPPERS = []
# ----------------------------------------------------------------------------#
# Terrain Options
# ----------------------------------------------------------------------------#
# Attributes for x will be called length, y width and z height
_C.TERRAIN = CN()

# Size of the "floor/0" x, y, z
_C.TERRAIN.SIZE = [25, 20, 1]

_C.TERRAIN.START_FLAT = 2

_C.TERRAIN.CENTER_FLAT = 2

# Supported types of terrain obstacles
_C.TERRAIN.TYPES = ["curve_slope"]

# Length of flat terrain
_C.TERRAIN.FLAT_LENGTH_RANGE = [1, 3, 1]

# Shared across avoid and jump
_C.TERRAIN.WALL_LENGTH = 0.1

# Length of terrain on which there will be hfield
_C.TERRAIN.HFIELD_LENGTH_RANGE = [4, 8, 4]

# Max height in case of slope profile
_C.TERRAIN.CURVE_HEIGHT_RANGE = [0.6, 1.2, 0.1]

_C.TERRAIN.BOUNDARY_WALLS = True

# Height of individual step
_C.TERRAIN.STEP_HEIGHT = 0.2

# Length of terrain on which there will be steps
_C.TERRAIN.STEP_LENGTH_RANGE = [12, 16, 4]

_C.TERRAIN.NUM_STEPS = 8

_C.TERRAIN.RUGGED_SQUARE_CLIP_RANGE = [0.2, 0.3, 0.1]

# Max height of bumps in bowl
_C.TERRAIN.BOWL_MAX_Z = 3.0

# Distance from origin where goal is placed
_C.TERRAIN.PATROL_HALF_LEN = 3.0

# Angle of incline for incline task
_C.TERRAIN.INCLINE_ANGLE = 0
# ----------------------------------------------------------------------------#
# Objects Options
# ----------------------------------------------------------------------------#
# Attributes for x will be called length, y width and z height
_C.OBJECT = CN()

# Goal position, if empty each episode will have a different goal position. Or
# you can specify the position here. Only specify the x, y position.
_C.OBJECT.GOAL_POS = []

# Same as GOAL_POS
_C.OBJECT.BOX_POS = []

# Min distance from the walls to place the object
_C.OBJECT.PLACEMENT_BUFFER_LEN = 3

_C.OBJECT.PLACEMENT_BUFFER_WIDTH = 5

# Half len of square for close placement
_C.OBJECT.CLOSE_PLACEMENT_DIST = 10

# Min distance between agent and goal for success
_C.OBJECT.SUCCESS_MARGIN = 0.5

# Side len of the box
_C.OBJECT.BOX_SIDE = 0.5

# Number of obstacles for obstacle env
_C.OBJECT.NUM_OBSTACLES = 20

# Length of the obstacle box
_C.OBJECT.OBSTACLE_LEN_RANGE = [0.5, 3, 0.5]

# Width of the obstacle box
_C.OBJECT.OBSTACLE_WIDTH_RANGE = [0.5, 3, 0.5]

# Range of distance between successive object placements for forward_placement
_C.OBJECT.FORWARD_PLACEMENT_DIST = [15, 25]

# Typpe of object to manipulate can be box or ball
_C.OBJECT.TYPE = "box"

_C.OBJECT.BALL_RADIUS = 0.15

_C.OBJECT.BOX_MASS = 1.0

# ----------------------------------------------------------------------------#
# Hfield Options
# ----------------------------------------------------------------------------#
_C.HFIELD = CN()

# For planer walker type unimals 1 otherwise 2
_C.HFIELD.DIM = 2

# Slice of hfield given to agent as obs. [behind, front, right, left] or
# [-x, +x, -y, +y]
_C.HFIELD.OBS_SIZE = [1, 4, 4, 4]

_C.HFIELD.ADAPTIVE_OBS = False

# See _cal_hfield_bounds in hfield.py
_C.HFIELD.ADAPTIVE_OBS_SIZE = [0.10, 0.50, 1.5, 5.0]

# Pad hfiled for handling agents on edges of the terrain. Padding value should
# be greater than sqrt(2) * max(HFIELD.OBS_SIZE). As when you rotate the
# the hfield obs, square diagonal should fit inside padding.
_C.HFIELD.PADDING = 10

# Number representing that the terrain has gap in hfield obs
_C.HFIELD.GAP_DEPTH = -10

# Number of divisions in 1 unit for hfield, should be a multiple of 10
_C.HFIELD.NUM_DIVS = 10

# Viz the extreme points of hfield
_C.HFIELD.VIZ = False

# ----------------------------------------------------------------------------#
# Image Options
# ----------------------------------------------------------------------------#
_C.IMAGE = CN()

# Save video
_C.IMAGE.SAVE = False

# Frame width
_C.IMAGE.WIDTH = 1920

# Frame height
_C.IMAGE.HEIGHT = 1080

# Camera
_C.IMAGE.CAMERA = "side"

# --------------------------------------------------------------------------- #
# PPO Options
# --------------------------------------------------------------------------- #
_C.PPO = CN()

# Discount factor for rewards
_C.PPO.GAMMA = 0.99

# GAE lambda parameter
_C.PPO.GAE_LAMBDA = 0.95

# Hyperparameter which roughly says how far away the new policy is allowed to
# go from the old
_C.PPO.CLIP_EPS = 0.2

# Number of epochs (K in PPO paper) of sgd on rollouts in buffer
_C.PPO.EPOCHS = 4

# Batch size for sgd (M in PPO paper)
_C.PPO.BATCH_SIZE = 512

# Value (critic) loss term coefficient
_C.PPO.VALUE_COEF = 0.5

# If KL divergence between old and new policy exceeds KL_TARGET_COEF * 0.01
# stop updates. Default value is high so that it's not used by default.
_C.PPO.KL_TARGET_COEF = 20.0

# Clip value function
_C.PPO.USE_CLIP_VALUE_FUNC = True

# Entropy term coefficient
_C.PPO.ENTROPY_COEF = 0.01

# Max timesteps per rollout
_C.PPO.TIMESTEPS = 128

# Number of parallel envs for collecting rollouts
_C.PPO.NUM_ENVS = 32

# Base learning rate (same for value and policy network)
_C.PPO.BASE_LR = 3e-4

# EPS for Adam/RMSProp
_C.PPO.EPS = 1e-5

# Use a linear schedule on the learning rate
_C.PPO.LINEAR_LR_DECAY = True

# Value to clip the gradient via clip_grad_norm_
_C.PPO.MAX_GRAD_NORM = 0.5

# Total number of env.step() across all processes and all rollouts over the
# course of training
_C.PPO.MAX_STATE_ACTION_PAIRS = 5e6

# Iter here refers to 1 cycle of experience collection and policy update.
# Refer PPO paper. This is field is inferred see: calculate_max_iters()
_C.PPO.MAX_ITERS = -1

# Length of video to save while evaluating policy in num env steps. Env steps
# may not be equal to actual simulator steps. Actual simulator steps would be
# env_steps * frame_skip.
_C.PPO.VIDEO_LENGTH = 1000

# XML for mujoco env
_C.PPO.XML_PATH = ""

# Path to load model from
_C.PPO.CHECKPOINT_PATH = ""

# --------------------------------------------------------------------------- #
# Model Options
# --------------------------------------------------------------------------- #
_C.MODEL = CN()

# Hidden dims for both actor/critic proprioceptive obs
_C.MODEL.PRO_HIDDEN_DIMS = [64, 64]

# Hidden dims for both actor/critic exterioceptive obs
_C.MODEL.EX_HIDDEN_DIMS = [64, 64]

# Hidden dims for combining proprioceptive and exterioceptive obs
_C.MODEL.OBS_FEAT_HIDDEN_DIMS = [64]
# --------------------------------------------------------------------------- #
# Sampler (VecEnv) Options
# --------------------------------------------------------------------------- #
_C.VECENV = CN()

# Type of vecenv. DummyVecEnv is generally the fastest option for light weight
# envs.
_C.VECENV.TYPE = "SubprocVecEnv"

# Number of envs to run in series for SubprocVecEnv
_C.VECENV.IN_SERIES = 4

# --------------------------------------------------------------------------- #
# Evolution Options
# --------------------------------------------------------------------------- #
_C.EVO = CN()

# Evolution without tournament phase is just random search. Use this to compare
# against random graph search.
_C.EVO.IS_EVO = True

# Number of prallel searches per node
_C.EVO.NUM_PROCESSES = 18

# Population size
_C.EVO.INIT_POPULATION_SIZE = 576

_C.EVO.INIT_METHOD = "limb_count_pop_init"

# Total number of unimals evolved over the course of evolution. Includes the
# initial population size. Note we can also do it on the basis of time but
# time can vary depending on implementation, hardware, etc.
_C.EVO.SEARCH_SPACE_SIZE = 4000

# Types of tournament selection: vanila (select N, choose best and remove the
# others), aging (select N from the most recent EVO.AGING_WINDOW_SIZE). _num
# means use EVO.NUM_PARTICIPANTS, _percent means use EVO.PERCENT_PARTICIPANTS.
_C.EVO.TOURNAMENT_TYPE = "aging_num"

# See EVO.TOURNAMENT_TYPE
_C.EVO.AGING_WINDOW_SIZE = 576

# Number of unimals in a tournament
_C.EVO.NUM_PARTICIPANTS = 4

# Percent of the active population to use during tournament selection.
# Alternative to EVO.NUM_PARTICIPANTS. In case of vanila tournament active
# population will be the percent of unimals currenlty alive (i.e number of
# metadata files). For aging tournament it will be a percentage of
# AGING_WINDOW_SIZE.
_C.EVO.PERCENT_PARTICIPANTS = 5

_C.EVO.MUTATION_OPS = [
    "grow_limb",
    "joint_angle",
    "limb_params",
    "dof",
    "density",
    "gear",
    "delete_limb"
]

_C.EVO.SELECTION_CRITERIA = ["__reward__forward", "__reward__stand"]

# Specify the objective corresponding to the criteria above. e.g you might want
# to minimize energy and maximize speed. -1 means maximize and +1 means
# minimize.
_C.EVO.SELECTION_CRITERIA_OBJ = [-1, -1]

# --------------------------------------------------------------------------- #
# CUDNN options
# --------------------------------------------------------------------------- #
_C.CUDNN = CN()

_C.CUDNN.BENCHMARK = False
_C.CUDNN.DETERMINISTIC = True

# ----------------------------------------------------------------------------#
# Misc Options
# ----------------------------------------------------------------------------#
# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries. This is the only seed
# which will effect env variations.
_C.RNG_SEED = 1

# Name of the environment used for experience collection
_C.ENV_NAME = "Hopper-v3"

# Use GPU
_C.USE_GPU = False

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters.
_C.LOG_PERIOD = -1

# Checkpoint period in iters. Refer LOG_PERIOD for meaning of iter
_C.CHECKPOINT_PERIOD = 100

# Evaluate the policy after every EVAL_PERIOD iters
_C.EVAL_PERIOD = -1

# Node ID for distributed runs
_C.NODE_ID = -1

# Number of nodes
_C.NUM_NODES = 1

# Unimal template path relative to the basedir
_C.UNIMAL_TEMPLATE = "./derl/envs/assets/unimal.xml"


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_default_cfg():
    return copy.deepcopy(cfg)
