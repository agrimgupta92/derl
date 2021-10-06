# Custom cameras that may be added to the arena for particular tasks.

# Camera config for final ft videos
FT_VIEW = {
    "name": "ft_view",
    "pos": [-4, -6, 2],
    "xyaxes": [1, -1, 0, 0, 1, 2],
    "mode": "trackcom",
}

# Camera config for final vt videos
VT_VIEW = {
    "name": "vt_view",
    "pos": [-1, -7, 2],
    "xyaxes": [1, -0.25, 0, 0, 0.75, 2],
    "mode": "trackcom",
}

# Camera config for final vt videos
PATROL_VIEW = {
    "name": "patrol_view",
    "pos": [0, -8, 5],
    "xyaxes": [1, 0, 0, 0, 1, 2],
    "mode": "fixed",
    "fovy": "50",
}

OBSTACLE_VIEW = {
    "name": "obstacle_view",
    "pos": [-9, -1.09046514, 7.51348819],
    "quat": [0.5953303, 0.3934467, -0.37115008, -0.5941626],
    "mode": "trackcom",
}

INCLINE_VIEW = {
    "name": "incline_view",
    "pos": [-1, -10, 2],
    "xyaxes": [1, -0.25, 0, 0, 0.75, 2],
    "mode": "trackcom",
}

MANI_VIEW = {
    "name": "mani_view",
    "pos": [-7.65528549, -0.24869655,  4.65546087],
    "quat": [0.5949708,  0.3811095 , -0.38163278, -0.59592086],
    "mode": "trackcom",
}

LEFT_VIEW = {
    "name": "left_view",
    "pos": [0.0, -6, 6],
    "xyaxes": [1.0, 0.0, 0.0, 0.0, 0.7, 0.75],
    "mode": "trackcom",
}

TOP_DOWN = {
    "name": "top_down",
    "pos": [0.0, 0.0, 20],
    "xyaxes": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "mode": "trackcom",
}

# Camera is behind unimal head (-x)
FRONT_VIEW = {
    "name": "front_view",
    "pos": [-6, 0.0, 6],
    "xyaxes": [0.0, -1.0, 0.0, 0.7, 0.0, 0.75],
    "mode": "trackcom",
}

# Camera is in front of unimal head (-x)
REAR_VIEW = {
    "name": "rear_view",
    "pos": [6, 0.0, 6],
    "xyaxes": [0.0, 1.0, 0.0, -0.7, 0.0, 0.75],
    "mode": "trackcom",
}

# Use this camera for tunning
TUNE_CAMERA = {
    "name": "tune_camera",
    "pos": [6, 0.0, 6],
    "xyaxes": [0.0, 1.0, 0.0, -0.7, 0.0, 0.75],
    "mode": "fixed",
}
