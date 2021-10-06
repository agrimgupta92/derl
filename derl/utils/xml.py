import numpy as np
from lxml import etree


def arr2str(array, num_decimals=-1):
    """Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"
    """
    if num_decimals > -1:
        arr = [round(a, num_decimals) for a in array]
    else:
        arr = array
    return " ".join(["{}".format(x) for x in arr])


def str2arr(string):
    """Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(" ")])


def add_list(list1, list2):
    assert len(list1) == len(list2)
    return [l1 + l2 for (l1, l2) in zip(list1, list2)]


def is_same_pos(arr1, arr2):
    """Check if two positions (x, y, z) are equal."""
    # Check if they are close up to eps. Needed as conversion from one
    # coordinate to another might introduce some rounding errors.
    diff = [abs(a1 - a2) for (a1, a2) in zip(arr1, arr2)]
    diff = sum(diff)
    if diff <= 0.03:
        return True

    return False


def axis2arr(axis):
    if axis == "x":
        return [-1, 0, 0]
    elif axis == "y":
        return [0, -1, 0]
    elif axis == "z":
        return [0, 0, -1]


"""Saving and loading XML as etree."""


def save_etree_as_xml(tree, path):
    """Save etree.ElementTree as xml file."""
    tree.write(
        path, xml_declaration=True, encoding="utf-8", pretty_print=True,
    )
    # Horrible!!! But this ensures that saved xml is always pretty
    root, tree = etree_from_xml(path)
    tree.write(
        path, xml_declaration=True, encoding="utf-8", pretty_print=True,
    )


def etree_from_xml(xml, ispath=True):
    """Load xml as etree and return root and tree."""
    if ispath:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.parse(xml, parser).getroot()
    else:
        root = etree.fromstring(xml)
    tree = etree.ElementTree(root)
    return root, tree


def etree_to_str(elem):
    """Convert etree elem to string."""
    return etree.tostring(elem, encoding="unicode", pretty_print=True)


"""Create XML elements corresponsing to mujoco."""


def site_elem(name, pos, s_class, size=None, fromto=None, type_=None):
    """Helper function to create site element."""
    attr_dict = {"name": name, "class": s_class}
    if pos:
        attr_dict["pos"] = arr2str(pos)
    if size:
        attr_dict["size"] = size
    if fromto:
        attr_dict["fromto"] = fromto
    if type_:
        attr_dict["type"] = type_
    return etree.Element("site", attr_dict)


def actuator_elem(name, gear):
    """Helper function to create motor actuator element."""

    return etree.Element(
        "motor", {"joint": name, "gear": "{}".format(gear), "name": name},
    )


def joint_elem(name, j_type, j_class, axis=None, range_=None, pos=None):
    """Helper function to create joint element."""
    attr_dict = {
        "name": name,
        "type": j_type,
        "class": j_class,
    }
    if range_:
        attr_dict["range"] = range_
    if pos:
        attr_dict["pos"] = pos
    if axis:
        attr_dict["axis"] = arr2str(axis)
    return etree.Element("joint", attr_dict)


def body_elem(name, pos):
    """Helper function to create body element."""
    assert isinstance(pos, list)
    return etree.Element("body", {"name": name, "pos": arr2str(pos)})


def sensor_elem(type_, name, site):
    """Helper function to create sensor element."""

    return etree.Element(type_, {"name": name, "site": site})


def camera_elem(camera_spec):
    camera_spec = camera_spec.copy()
    camera_spec["pos"] = arr2str(camera_spec["pos"])
    if "xyaxes" in camera_spec:
        camera_spec["xyaxes"] = arr2str(camera_spec["xyaxes"])
    elif "quat" in camera_spec:
        camera_spec["quat"] = arr2str(camera_spec["quat"])
    elif "fovy" in camera_spec:
        camera_spec["fovy"] = arr2str(camera_spec["fovy"])
    return etree.Element("camera", camera_spec)


def floor_segm(
    name, pos, size, geom_type, material="grid", density=None, incline=None
):
    """Helper function to create floor segments."""
    pos = [round(_, 2) for _ in pos]
    attrs = {
        "name": name,
        "pos": arr2str(pos),
        "type": geom_type,
        "material": material,
    }

    if size:
        size = [round(_, 2) for _ in size]
        attrs["size"] = arr2str(size)
    if geom_type == "hfield":
        attrs["hfield"] = name
    if density:
        attrs["density"] = str(density)
    if incline:
        attrs["euler"] = arr2str([0, incline, 0])
    return etree.Element("geom", attrs)


def exclude_elem(name1, name2):
    """Exclude contact between geom of name 1 and 2."""
    attrs = {
        "name": "{}:{}".format(name1, name2),
        "body1": name1,
        "body2": name2
    }
    return etree.Element("exclude", attrs)


def hfield_asset(name, nrow, ncol, size):
    """Helper function to create hfield segments."""
    size = [round(_, 2) for _ in size]
    return etree.Element(
        "hfield",
        {
            "name": name,
            "nrow": str(nrow),
            "ncol": str(ncol),
            "size": arr2str(size),
        },
    )


def find_elem(etree_elem, tag, attr_type=None, attr_value=None, child_only=False):
    if child_only:
        xpath = "./"
    else:
        xpath = ".//"

    if attr_type:
        return [
            tag
            for tag in etree_elem.iterfind(
                '{}{}[@{}="{}"]'.format(xpath, tag, attr_type, attr_value)
            )
        ]
    else:
        return [tag for tag in etree_elem.iterfind("{}{}".format(xpath, tag))]


def name2id(elem):
    """Returns id of the elem."""
    elem_name = elem.get("name")
    return int(elem_name.split("/")[-1])
