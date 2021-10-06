import derl.utils.xml as xu
from derl.config import cfg
from derl.envs.modules.agent import extract_agent_from_xml
from derl.envs.modules.agent import merge_agent_with_base
from derl.envs.tasks.escape_bowl import make_env_escape_bowl
from derl.envs.tasks.exploration import make_env_exploration
from derl.envs.tasks.incline import make_env_incline
from derl.envs.tasks.locomotion import make_env_locomotion
from derl.envs.tasks.manipulation import make_env_manipulation
from derl.envs.tasks.obstacle import make_env_obstacle
from derl.envs.tasks.patrol import make_env_patrol
from derl.envs.tasks.point_nav import make_env_point_nav
from derl.envs.tasks.push_box_incline import make_env_push_box_incline
from derl.envs.wrappers.select_keys import SelectKeysWrapper
from derl.utils import file as fu


def modify_xml_attributes(xml):
    root, tree = xu.etree_from_xml(xml, ispath=False)

    # Modify njmax and nconmax
    size = xu.find_elem(root, "size")[0]
    size.set("njmax", str(cfg.XML.NJMAX))
    size.set("nconmax", str(cfg.XML.NCONMAX))

    # Enable/disable filterparent
    flag = xu.find_elem(root, "flag")[0]
    flag.set("filterparent", str(cfg.XML.FILTER_PARENT))

    # Modify default geom params
    default_geom = xu.find_elem(root, "geom")[0]
    default_geom.set("condim", str(cfg.XML.GEOM_CONDIM))
    default_geom.set("friction", xu.arr2str(cfg.XML.GEOM_FRICTION))

    # Modify njmax and nconmax
    visual = xu.find_elem(root, "visual")[0]
    map_ = xu.find_elem(visual, "map")[0]
    map_.set("shadowclip", str(cfg.XML.SHADOWCLIP))

    return xu.etree_to_str(root)


def make_env(xml_path=None):
    if xml_path:
        agent = xml_path
        agent = extract_agent_from_xml(agent)
        ispath = False
        unimal_id = fu.path2id(xml_path)
    else:
        agent = cfg.ENV.WALKER
        ispath = True
        unimal_id = ""

    xml = merge_agent_with_base(agent, ispath)
    xml = modify_xml_attributes(xml)

    env_func = "make_env_{}".format(cfg.ENV.TASK)
    env = globals()[env_func](xml, unimal_id)

    # Add common wrappers in the end
    env = SelectKeysWrapper(env, keys_to_keep=cfg.ENV.KEYS_TO_KEEP)
    return env
