import argparse
import sys

from derl.algos.ppo.envs import make_env
from derl.config import cfg
from derl.envs.env_viewer import EnvViewer


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Create random morphologies.")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.ENV.WALKER = "./derl/envs/assets/unimal_ant.xml"
    cfg.freeze()

    env = make_env(cfg.ENV_NAME, cfg.RNG_SEED, 1)()
    env_viewer = EnvViewer(env)
    env_viewer.run()


if __name__ == "__main__":
    main()
