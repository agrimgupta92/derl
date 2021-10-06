import itertools

from lxml import etree
from mujoco_py import MjSim
from mujoco_py import load_model_from_xml


def mj_name2id(sim, type_, name):
    """Returns the mujoco id corresponding to name."""
    if type_ == "site":
        return sim.model.site_name2id(name)
    elif type_ == "geom":
        return sim.model.geom_name2id(name)
    elif type_ == "body":
        return sim.model.body_name2id(name)
    elif type_ == "sensor":
        return sim.model.sensor_name2id(name)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mj_id2name(sim, type_, id_):
    """Returns the mujoco name corresponding to id."""
    if type_ == "site":
        return sim.model.site_id2name(id_)
    elif type_ == "geom":
        return sim.model.geom_id2name(id_)
    elif type_ == "body":
        return sim.model.body_id2name(id_)
    elif type_ == "sensor":
        return sim.model.sensor_id2name(id_)
    else:
        raise ValueError("type_ {} is not supported.".format(type_))


def mjsim_from_etree(root):
    """Return MjSim from etree root."""
    return MjSim(mjmodel_from_etree(root))


def mjmodel_from_etree(root):
    """Return MjModel from etree root."""
    model_string = etree.tostring(root, encoding="unicode", pretty_print=True)
    return load_model_from_xml(model_string)


def joint_qpos_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qpos values."""
    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qpos_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qpos values of all joints matching the prefix."""
    qpos_idxs_list = [
        joint_qpos_idxs(sim, name)
        for name in sim.model.joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def qpos_idxs_for_agent(sim):
    """Gets indexes for the qpos values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qpos_idxs_list = [joint_qpos_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qpos_idxs_list))


def joint_qvel_idxs(sim, joint_name):
    """Gets indexes for the specified joint's qvel values."""
    addr = sim.model.get_joint_qvel_addr(joint_name)
    if isinstance(addr, tuple):
        return list(range(addr[0], addr[1]))
    else:
        return [addr]


def qvel_idxs_from_joint_prefix(sim, prefix):
    """Gets indexes for the qvel values of all joints matching the prefix."""
    qvel_idxs_list = [
        joint_qvel_idxs(sim, name)
        for name in sim.model.joint_names
        if name.startswith(prefix)
    ]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def qvel_idxs_for_agent(sim):
    """Gets indexes for the qvel values of all agent joints."""
    agent_joints = names_from_prefixes(sim, ["root", "torso", "limb"], "joint")
    qvel_idxs_list = [joint_qvel_idxs(sim, name) for name in agent_joints]
    return list(itertools.chain.from_iterable(qvel_idxs_list))


def names_from_prefixes(sim, prefixes, elem_type):
    """Get all names of elem_type elems which match any of the prefixes."""
    all_names = getattr(sim.model, "{}_names".format(elem_type))
    matches = []
    for name in all_names:
        for prefix in prefixes:
            if name.startswith(prefix):
                matches.append(name)
                break
    return matches


def get_active_contacts(sim):
    num_contacts = sim.data.ncon
    contacts = sim.data.contact[:num_contacts]
    contact_geoms = [
        tuple(
            sorted(
                (
                    mj_id2name(sim, "geom", contact.geom1),
                    mj_id2name(sim, "geom", contact.geom2),
                )
            )
        )
        for contact in contacts
    ]
    return sorted(list(set(contact_geoms)))
