import numpy as np

from derl.config import cfg


def uniform_placement(grid, obj_size, random_state):
    nrow, ncol = grid.shape
    buffer_l = cfg.OBJECT.PLACEMENT_BUFFER_LEN * cfg.HFIELD.NUM_DIVS * 2
    buffer_w = cfg.OBJECT.PLACEMENT_BUFFER_WIDTH * cfg.HFIELD.NUM_DIVS * 2
    # Keep 1 meter + obj half len padding from the wall
    row_max = nrow - obj_size[1] - buffer_w
    col_max = ncol - obj_size[0] - buffer_l

    row_min = obj_size[1] + buffer_w
    col_min = obj_size[0] + buffer_l
    idx = np.array(
        [
            random_state.randint(row_min, row_max),
            random_state.randint(col_min, col_max),
        ]
    )
    return idx


def close_placement(center, grid, obj_size, random_state):
    # Place a object close to the position specified by center
    row_c, col_c = center
    nrow, ncol = grid.shape
    buffer_l = cfg.OBJECT.PLACEMENT_BUFFER_LEN * cfg.HFIELD.NUM_DIVS * 2
    buffer_w = cfg.OBJECT.PLACEMENT_BUFFER_WIDTH * cfg.HFIELD.NUM_DIVS * 2
    side_len = cfg.OBJECT.CLOSE_PLACEMENT_DIST * cfg.HFIELD.NUM_DIVS

    # Keep 1 meter + obj half len padding from the wall
    row_lim = nrow - obj_size[1] - buffer_w
    col_lim = ncol - obj_size[0] - buffer_l

    row_min = max(buffer_w, row_c - side_len)
    row_max = min(row_lim, row_c + side_len)
    col_min = max(buffer_l, col_c - side_len)
    col_max = min(col_lim, col_c + side_len)

    idx = np.array(
        [
            random_state.randint(row_min, row_max),
            random_state.randint(col_min, col_max),
        ]
    )
    return idx


def forward_placement(prev_pos, grid, obj_size, random_state):
    # Place a object close to the position specified by center
    row_c, col_c = prev_pos
    nrow, ncol = grid.shape
    buffer_l = cfg.OBJECT.PLACEMENT_BUFFER_LEN * cfg.HFIELD.NUM_DIVS * 2
    buffer_w = cfg.OBJECT.PLACEMENT_BUFFER_WIDTH * cfg.HFIELD.NUM_DIVS * 2
    lo_dist, hi_dist = cfg.OBJECT.FORWARD_PLACEMENT_DIST
    min_col_dist = random_state.randint(lo_dist, hi_dist) * cfg.HFIELD.NUM_DIVS

    # Keep 1 meter + obj half len padding from the wall
    row_max = nrow - obj_size[1] - buffer_w
    row_min = obj_size[1] + buffer_w

    col_idx = min(col_c + min_col_dist, ncol - obj_size[0] - buffer_l)

    idx = np.array(
        [random_state.randint(row_min, row_max), col_idx]
    )
    return idx


def place_on_grid(arena, obj_size, center=None, update_grid=True):
    grid = arena.placement_grid
    divs = cfg.HFIELD.NUM_DIVS
    obj_size_in_divs = np.ceil(obj_size * divs).astype(int)
    rnd_state = arena.np_random

    for i in range(10):
        if cfg.ENV.TASK in ["manipulation"]:
            row_idx, col_idx = forward_placement(
                center, grid, obj_size_in_divs, rnd_state
            )
        elif center:
            row_idx, col_idx = close_placement(
                center, grid, obj_size_in_divs, rnd_state
            )
        else:
            row_idx, col_idx = uniform_placement(
                grid, obj_size_in_divs, rnd_state
            )

        if np.any(
            grid[
                row_idx - obj_size_in_divs[1] : row_idx + obj_size_in_divs[1],
                col_idx - obj_size_in_divs[0] : col_idx + obj_size_in_divs[0],
            ]
        ):
            continue
        else:
            if update_grid:
                grid[
                    row_idx - obj_size_in_divs[1] : row_idx + obj_size_in_divs[1],
                    col_idx - obj_size_in_divs[0] : col_idx + obj_size_in_divs[0],
                ] = 1
            return arena.grid_idx_to_pos(np.asarray([row_idx, col_idx]))
    return None
