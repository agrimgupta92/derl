import os
import sys
from pathlib import Path

from derl.config import cfg
from derl.utils import file as fu


def handle_exception(err, custom_msg, unimal_id=None):
    """Print err and custom message, save a file marking err and exit proc."""
    print(custom_msg)
    print(err)

    if cfg.VECENV.TYPE == "SubprocVecEnv":
        proc_id = os.getppid()
    else:
        proc_id = os.getpid()

    process_end = os.path.join(cfg.OUT_DIR, "{}_{}".format(cfg.NODE_ID, proc_id))
    Path(process_end).touch()
    if unimal_id:
        error_path = fu.id2path(unimal_id, "error_metadata")
        Path(error_path).touch()
    sys.exit(1)
