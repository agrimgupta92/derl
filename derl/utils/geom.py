import numpy as np


def sph2cart(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates."""
    # We use the math defination of angles from wikipedia
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    point = [x, y, z]
    point = [round(coord, 2) for coord in point]
    for idx, _ in enumerate(point):
        if point[idx] == -0.0:
            point[idx] = abs(point[idx])
    return point


def cart2sph(x, y, z):
    """Convert cartesian coordinates to spherical coordinates."""
    r = np.sqrt(np.sum(np.square([x, y, z])))
    theta = np.arctan2(y, x)
    # arctan2 will return theta in range [-pi, pi]. We need theta in range
    # [0, 2 * pi].
    if theta < 0:
        theta = 2 * np.pi + theta
    phi = np.arccos(z / r)
    return [r, theta, phi]


def is_same_orient(o1, o2):
    # Ignore magnitude
    _, theta1, phi1 = o1
    _, theta2, phi2 = o2

    if theta1 == theta2 and phi1 == phi2:
        return True

    if phi1 == phi2 and phi1 in [0, np.pi]:
        return True

    return False


def angle_between(v1, v2):
    """Angle between 2D vectors."""
    assert len(v1) == len(v2)
    assert len(v1) == 2
    # Counter clock wise is +ve, clockwise -ve
    # Refer: https://stackoverflow.com/a/16544330
    dot = np.dot(v1, v2)
    det = np.linalg.det(np.stack((v1, v2)))
    return np.arctan2(det, dot)


def dir_a2b(a, b):
    """Return unit vector pointing from a to b."""
    dir_ = b - a
    return dir_ / np.linalg.norm(dir_)


def vec_a2b(a, b):
    """Return vector pointing from a to b."""
    return np.asarray(b) - np.asarray(a)


def normalize_vec(a):
    """Return unit vector in dir of a."""
    return a / np.linalg.norm(a)

