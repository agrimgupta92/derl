import hashlib
import itertools
import multiprocessing
from multiprocessing import Pool

import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from derl.utils import file as fu
from derl.utils import mjpy as mu
from derl.utils import xml as xu


def point_cloud_from_xml(path):
    """Create point cloud from unimal xml path."""
    unimal_id = fu.path2id(path)
    root, tree = xu.etree_from_xml(path)
    sim = mu.mjsim_from_etree(root)
    sim.forward()
    # Get sites which describe the limbs
    site_prefixes = ["limb/btm/", "torso", "limb/mid/"]
    sites = mu.names_from_prefixes(sim, site_prefixes, "site")
    # Get actual position of the sites
    sparse_point_cloud = []
    for site_name in sites:
        # Note this is necessary otherwise procs don't exit
        x, y, z = sim.data.get_site_xpos(site_name)
        pos = [x, y, z]
        pos = [round(_, 2) for _ in pos]
        sparse_point_cloud.append(pos)
    return [unimal_id, sparse_point_cloud]


def geom_orientations_from_xml(path):
    """Create a list of geom orientations."""
    unimal_id = fu.path2id(path)
    root, tree = xu.etree_from_xml(path)
    sim = mu.mjsim_from_etree(root)
    sim.forward()
    limbs = mu.names_from_prefixes(sim, ["limb/"], "geom")

    limb_orientations = []
    for limb_name in limbs:
        geom_frame = sim.data.get_geom_xmat(limb_name).copy().ravel()
        limb_orientations.append(geom_frame)

    return [unimal_id, limb_orientations]


def hash_from_xml(path):
    """Return hash of the xml file, most narrow form of comparision."""
    unimal_id = fu.path2id(path)
    root, tree = xu.etree_from_xml(path)
    contact = root.findall("./contact")[0]
    contact_pairs = xu.find_elem(contact, "exclude")
    for cp in contact_pairs:
        contact.remove(cp)

    assets = root.findall("./asset")
    for asset in assets:
        root.remove(asset)
    xml_string = xu.etree_to_str(root)
    return [unimal_id, hashlib.sha224(str.encode(xml_string)).hexdigest()]


def get_ancestor_from_xml(path):
    """Return the original ancestor for the unimal."""
    unimal_id = fu.path2id(path)
    metadata_path = fu.get_corresponding_folder_paths([path], "metadata")[0]
    metadata = fu.load_json(metadata_path)
    return [unimal_id, metadata["lineage"].split("/")[0]]


def get_metric_in_parallel(paths, metric_name):
    """Get similarity metric for a list of unimals."""
    p = Pool()
    if metric_name == "point_cloud":
        data = p.map(point_cloud_from_xml, paths)
    elif metric_name == "geom_orientation":
        data = p.map(geom_orientations_from_xml, paths)
    elif metric_name == "ancestor":
        data = p.map(get_ancestor_from_xml, paths)
    elif metric_name == "hash":
        data = p.map(hash_from_xml, paths)
    else:
        raise ValueError("Metric {} not supported.".format(metric_name))

    p.close()
    p.join()
    return {uid: m for uid, m in data}


def is_same_morphology(m1, m2):
    """Return True if unimals have same num_limbs and same metric for all limbs."""
    cost = cdist(m1, m2)
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment_cost = cost[row_ind, col_ind].sum()
    if len(m1[0]) == 9:
        eps = 0.2
    else:
        eps = 1e-3
    if assignment_cost < eps and len(m1) == len(m2):
        return True
    else:
        return False


def check_all_pair_sim(all_pairs, unimal_m):
    # Create all pairs metric
    all_pairs_pc = [[unimal_m[u1], unimal_m[u2]] for u1, u2 in all_pairs]
    p = Pool(int(multiprocessing.cpu_count() * 0.50))
    data = p.starmap(is_same_morphology, all_pairs_pc)
    p.close()
    p.join()
    return data


def check_all_pair_ancestry(all_pairs, unimal_ancestor):
    all_pairs_same_ancestry = []
    for uid1, uid2 in all_pairs:
        if unimal_ancestor[uid1] == unimal_ancestor[uid2]:
            all_pairs_same_ancestry.append(True)
        else:
            all_pairs_same_ancestry.append(False)
    return all_pairs_same_ancestry


def check_all_pair_same_individual(all_pairs, unimal_hash):
    all_pairs_same_ind = []
    for uid1, uid2 in all_pairs:
        if unimal_hash[uid1] == unimal_hash[uid2]:
            all_pairs_same_ind.append(True)
        else:
            all_pairs_same_ind.append(False)
    return all_pairs_same_ind


def create_graph(all_pairs, all_pairs_sim):
    # Create graph with nodes as unimal ids and edges between
    # them if they have the same morphology
    G = nx.Graph()
    for pair, pair_sim in zip(all_pairs, all_pairs_sim):
        if not pair_sim:
            G.add_nodes_from(list(pair))
            continue
        # If pair is same add edge to graph
        G.add_edge(*pair)

    return G


def create_graph_from_xml_paths(xml_paths, metric_name, graph_type):
    # Create dict {uid: point_cloud}
    unimal_m = get_metric_in_parallel(xml_paths, metric_name)
    # Create list of all unimal pairing
    all_pairs = list(itertools.combinations(unimal_m.keys(), 2))

    if graph_type == "individual":
        all_pairs_sim = check_all_pair_same_individual(all_pairs, unimal_m)
        return create_graph(all_pairs, all_pairs_sim)

    # Check if two unimals have the same morphology
    all_pairs_sim = check_all_pair_sim(all_pairs, unimal_m)
    if graph_type == "family":
        unimal_ancestor = get_metric_in_parallel(xml_paths, "ancestor")
        all_pairs_sim_a = check_all_pair_ancestry(all_pairs, unimal_ancestor)
        all_pairs_sim = [
            s1 or s2 for s1, s2 in zip(all_pairs_sim, all_pairs_sim_a)
        ]

    return create_graph(all_pairs, all_pairs_sim)


def create_graph_from_uids(
    sweep_name, uids, metric_name, graph_type="species", task_num=1
):
    xml_paths = [
        fu.id2path(uid, "xml", sweep_name=sweep_name, task_num=task_num)
        for uid in uids
    ]
    return create_graph_from_xml_paths(xml_paths, metric_name, graph_type)
