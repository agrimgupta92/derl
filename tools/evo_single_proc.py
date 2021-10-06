import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Torch 1.5 bug, need to import numpy before torch
# Refer: https://github.com/pytorch/pytorch/issues/37377
import numpy as np
import torch

from derl.algos.ppo.ppo import PPO
from derl.config import cfg
from derl.envs.morphology import SymmetricUnimal
from derl.utils import evo as eu
from derl.utils import exception as exu
from derl.utils import file as fu
from derl.utils import sample as su


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument("--proc-id", required=True, type=int)
    parser.add_argument(
        "opts",
        help="See morphology/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train(xml_file, id_, parent_metadata=None):
    su.set_seed(cfg.RNG_SEED, use_strong_seeding=False)
    # Setup torch
    torch.set_num_threads(1)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    # Train unimal
    PPOTrainer = PPO(xml_file=xml_file)

    exit_cond = "search_space"
    if parent_metadata is None:
        # Exit early if population already initialized
        exit_cond = "population_init"

    PPOTrainer.train(exit_cond=exit_cond)

    if (
        exit_cond == "population_init" and
        eu.get_population_size() >= cfg.EVO.INIT_POPULATION_SIZE
    ):
        return

    if (
        exit_cond == "search_space" and
        eu.get_searched_space_size() >= cfg.EVO.SEARCH_SPACE_SIZE
    ):
        return

    # Save the model
    PPOTrainer.save_model(path=fu.id2path(id_, "models"))
    # Save the rewards
    PPOTrainer.save_rewards(path=fu.id2path(id_, "rewards"))

    if eu.should_save_video():
        PPOTrainer.save_video(os.path.join(cfg.OUT_DIR, "videos"))

    # Create metadata to be used for next steps of evolution
    metadata = {}
    # Add mean of last N (100) values of rewards
    for rew_type, rews in PPOTrainer.mean_ep_rews.items():
        metadata[rew_type] = int(np.mean(rews[-100:]))

    metadata["pos"] = np.mean(PPOTrainer.mean_pos[-100:])

    metadata["id"] = "{}".format(id_)
    if not parent_metadata:
        metadata["lineage"] = "{}".format(id_)
    else:
        metadata["lineage"] = "{}/{}".format(parent_metadata["lineage"], id_)

    # Save metadata to disk
    path = os.path.join(fu.get_subfolder("metadata"), "{}.json".format(id_))
    fu.save_json(metadata, path)


def init_done(unimal_id):
    unimal_idx = int(unimal_id.split(".")[0].split("-")[1])
    success_metadata = fu.get_files(fu.get_subfolder("metadata"), ".*json")
    error_metadata = fu.get_files(fu.get_subfolder("error_metadata"), ".*json")
    done_metadata = success_metadata + error_metadata
    done_idx = [
        int(path.split("/")[-1].split("-")[2].split(".")[0])
        for path in done_metadata
    ]
    if unimal_idx in done_idx:
        return True
    else:
        return False


def init_population(proc_id):
    init_done_path = os.path.join(cfg.OUT_DIR, "init_pop_done")

    if os.path.isfile(init_done_path):
        print("Population has already been initialized.")
        return

    # Divide work by num nodes and then num procs
    xml_paths = fu.get_files(
        fu.get_subfolder("xml"), ".*xml", sort=True, sort_type="time"
    )[: cfg.EVO.INIT_POPULATION_SIZE]
    xml_paths.sort()
    xml_paths = fu.chunkify(xml_paths, cfg.NUM_NODES)[cfg.NODE_ID]
    xml_paths = fu.chunkify(xml_paths, cfg.EVO.NUM_PROCESSES)[proc_id]

    for xml_path in xml_paths:
        unimal_id = fu.path2id(xml_path)

        if init_done(unimal_id):
            print("{} already done, proc_id: {}".format(unimal_id, proc_id))
            continue

        ppo_train(fu.id2path(unimal_id, "xml"), unimal_id)

        if eu.get_population_size() >= cfg.EVO.INIT_POPULATION_SIZE:
            break

    # Explicit file is needed as current population size can be less than
    # initial population size. In fact after the first round of tournament
    # selection population size can be as low as half of
    Path(init_done_path).touch()


def tournament_evolution(idx):
    seed = cfg.RNG_SEED + (cfg.NODE_ID * cfg.EVO.NUM_PROCESSES + idx) * 100
    while eu.get_searched_space_size() < cfg.EVO.SEARCH_SPACE_SIZE:
        su.set_seed(seed, use_strong_seeding=True)
        seed += 1
        parent_metadata = eu.select_parent()
        child_id = "{}-{}-{}".format(
            cfg.NODE_ID, idx, datetime.now().strftime("%d-%H-%M-%S")
        )
        unimal = SymmetricUnimal(
            child_id, init_path=fu.id2path(parent_metadata["id"], "unimal_init"),
        )
        unimal.mutate()
        unimal.save()

        ppo_train(fu.id2path(child_id, "xml"), child_id, parent_metadata)

    # Even though video meta files are removed inside ppo, sometimes it might
    # fail in between creating video. In such cases, we just remove the video
    # metadata file as master proc uses absence of meta files as sign of completion.
    video_dir = fu.get_subfolder("videos")
    video_meta_files = fu.get_files(
        video_dir, "{}-{}-.*json".format(cfg.NODE_ID, idx)
    )
    for video_meta_file in video_meta_files:
        fu.remove_file(video_meta_file)


def evolve_single_proc(idx):
    init_population(idx)
    if cfg.EVO.IS_EVO:
        tournament_evolution(idx)


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    # Unclear why this happens, very rare
    if cfg.OUT_DIR == "/tmp":
        exu.handle_exception("", "ERROR TMP")

    evolve_single_proc(args.proc_id)
    print("Node ID: {}, Proc ID: {} finished.".format(cfg.NODE_ID, args.proc_id))


if __name__ == "__main__":
    main()
