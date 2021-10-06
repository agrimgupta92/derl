import random

import gym
import numpy as np
import torch


def sample_orient(r, step_size=45):
    """Random orientation in spherical coordinates.

    Returns:
        list[float]: A vector of length r in lower hemisphere.
    """
    if step_size not in [30, 45]:
        raise ValueError("step_size {} not supported.".format(step_size))

    # Theta is angle from x axis. 0 <= theta <= 360 - step_size
    theta = np.radians(np.arange(360 / step_size) * step_size)
    # Phi is angle from z axis. 90 <= phi <= 180
    phi = np.radians(np.arange((90 / step_size) + 1) * step_size + 90)
    theta = np.random.choice(theta, 1)[0]
    phi = np.random.choice(phi, 1)[0]
    return [r, theta, phi]


def sample_from_range(range_, rng_state=None):
    """Randomly sample a value from the list specified by range_."""

    if len(range_) == 1:
        return range_[0]

    assert len(range_) == 3

    # To make the end point inclusive
    list_ = np.arange(*range_)
    list_ = [round(_, 2) for _ in list_]
    # arange does not handle endpoints nicely for < 1 step size. Add endpoint
    # manually.
    list_.append(range_[1])
    if rng_state:
        return rng_state.choice(list_, 1)[0]
    else:
        return np.random.choice(list_, 1)[0]


def sample_from_list(list_, rng_state=None):
    """Randomly sample a element of the list."""
    if rng_state:
        return rng_state.choice(list_, 1)[0]
    else:
        return np.random.choice(list_, 1)[0]


def sample_range_from_range(range_):
    """Randomly sample (low, hi) from the list specified by range_."""

    if len(range_) == 1:
        val = abs(range_[0])
        return [-val, val]

    assert len(range_) == 3

    while True:
        # For low last endpoint is not included
        list_ = np.arange(*range_)
        low = np.random.choice(list_, 1)[0]

        list_ = list(np.arange(low, range_[1], range_[2]))
        list_.append(range_[1])
        # low should be removed
        list_ = list_[1:]
        if len(list_) == 0:
            continue
        hi = np.random.choice(list_, 1)[0]
        return [round(low, 2), round(hi, 2)]


def sample_joint_angle(range_):

    if len(range_) == 1:
        val = abs(range_[0])
        return [-val, val]

    assert len(range_) == 3
    assert range_[0] == 0

    while True:
        list_ = list(range(*range_))
        list_.append(range_[1])

        low, hi = random.choices(list_, k=2)
        if low == hi and low == 0:
            continue
        if (-low, hi) == (-30, 0) or (-low, hi) == (0, 30):
            continue
        return [-low, hi]


def sample_joint_angle_from_list(list_):
    return random.choice(list_)


def set_seed(seed, idx=0, use_strong_seeding=False):
    seed = seed + idx
    if use_strong_seeding:
        seed = strong_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def strong_seed(seed):
    """Get a strong uncorrelated seed from naive seeding."""
    seed = gym.utils.seeding.hash_seed(seed)
    _, seed = divmod(seed, 2 ** 32)
    return seed
