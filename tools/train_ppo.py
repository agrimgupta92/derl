import argparse
import os
import sys

import torch

from derl.algos.ppo.ppo import PPO
from derl.config import cfg
from derl.config import dump_cfg
from derl.utils import sample as su


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS)
        // cfg.PPO.TIMESTEPS
        // cfg.PPO.NUM_ENVS
    )


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    su.set_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    PPOTrainer = PPO(xml_file=cfg.PPO.XML_PATH)
    PPOTrainer.train()
    if PPOTrainer.is_master_proc():
        PPOTrainer.save_rewards()
        PPOTrainer.save_model()
        # PPOTrainer.save_video(cfg.OUT_DIR)


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # Infer OPTIM.MAX_ITERS
    calculate_max_iters()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg()
    ppo_train()


if __name__ == "__main__":
    main()
