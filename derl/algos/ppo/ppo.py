import os
import time
from collections import defaultdict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from derl.config import cfg
from derl.envs.vec_env.vec_video_recorder import VecVideoRecorder
from derl.utils import evo as eu
from derl.utils import file as fu

from .buffer import Buffer
from .envs import get_ob_rms
from .envs import make_vec_envs
from .envs import set_ob_rms
from .model import ActorCritic
from .model import Agent


class PPO:
    def __init__(self, xml_file=None):
        # Create vectorized envs
        self.envs = make_vec_envs(xml_file=xml_file)
        self.xml_file = xml_file
        # File prefix for saving
        if xml_file:
            self.file_prefix = xml_file.split("/")[-1].split(".")[0]
        else:
            self.file_prefix = cfg.ENV_NAME

        self.device = torch.device("cuda:0" if cfg.USE_GPU else "cpu")
        # Setup actor, critic
        self.actor_critic = ActorCritic(
            self.envs.observation_space, self.envs.action_space
        )

        self.actor_critic.to(self.device)
        self.agent = Agent(self.actor_critic)

        # Setup experience buffer
        self.buffer = Buffer(
            self.envs.observation_space, self.envs.action_space.shape
        )
        # Optimizer for both actor and critic
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS
        )

        # Stores mean episode rewards, also stores individual components of
        # rewards like standing reward etc.
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        # Human interpretable metric for each task. Each task should have one!
        self.mean_metric = []

        self.fps = 0

    def train(self, exit_cond=None):
        obs = self.envs.reset()
        ep_rew = defaultdict(lambda: deque(maxlen=10))
        ep_pos = deque(maxlen=10)
        ep_vel = deque(maxlen=10)
        ep_metric = deque(maxlen=10)
        self.start = time.time()

        for cur_iter in range(cfg.PPO.MAX_ITERS):

            if cfg.PPO.LINEAR_LR_DECAY:
                # Decrease learning rate linearly
                lr_linear_decay(
                    self.optimizer, cur_iter, cfg.PPO.MAX_ITERS, cfg.PPO.BASE_LR
                )

            for step in range(cfg.PPO.TIMESTEPS):
                # Sample actions
                val, act, logp = self.agent.act(obs)

                next_obs, reward, done, infos = self.envs.step(act)

                for info in infos:
                    if "episode" in info.keys():
                        ep_rew["reward"].append(info["episode"]["r"])

                        for rew_type, rew_ in info["episode"].items():
                            if "__reward__" in rew_type:
                                ep_rew[rew_type].append(rew_)

                        if "x_pos" in info:
                            ep_pos.append(info["x_pos"])
                        if "x_vel" in info:
                            ep_vel.append(info["x_vel"])
                        if "metric" in info:
                            ep_metric.append(info["metric"])

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                ).to(self.device)
                timeouts = torch.FloatTensor(
                    [
                        [0.0] if "timeout" in info.keys() else [1.0]
                        for info in infos
                    ]
                ).to(self.device)

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts)
                obs = next_obs

            next_val = self.agent.get_value(obs)
            self.buffer.compute_returns(next_val)
            self.train_on_batch()

            if (
                cur_iter % cfg.LOG_PERIOD == 0
                and cfg.LOG_PERIOD > 0
                and len(ep_rew["reward"]) > 1
            ):
                self._log_stats(cur_iter, ep_rew["reward"], np.mean(ep_metric))
                self.save_model()

            if len(ep_pos) > 1:
                for rew_type, rews_ in ep_rew.items():
                    self.mean_ep_rews[rew_type].append(round(np.mean(rews_), 2))
                self.mean_pos.append(round(np.mean(ep_pos), 2))

            if len(ep_vel) > 1:
                self.mean_vel.append(round(np.mean(ep_vel), 2))
            if len(ep_metric) > 1:
                self.mean_metric.append(round(np.mean(ep_metric), 2))

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

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for _ in range(cfg.PPO.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)

            for batch in batch_sampler:
                # Reshape to do in a single forward pass for all steps
                val, _, logp, ent = self.actor_critic(batch["obs"], batch["act"])
                clip_ratio = cfg.PPO.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])
                approx_kl = (batch["logp_old"] - logp).mean().item()
                if approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01:
                    return

                surr1 = ratio * batch["adv"]

                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                surr2 *= batch["adv"]

                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.PPO.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(
                        -clip_ratio, clip_ratio
                    )
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                self.optimizer.zero_grad()

                loss = val_loss * cfg.PPO.VALUE_COEF
                loss += pi_loss
                loss += -ent * cfg.PPO.ENTROPY_COEF
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM
                )
                self.optimizer.step()

    def save_model(self, path=None):
        if not path:
            path = os.path.join(cfg.OUT_DIR, self.file_prefix + ".pt")
        torch.save([self.actor_critic, get_ob_rms(self.envs)], path)

    def _log_stats(self, cur_iter, ep_rew, ep_metric):
        self._log_fps(cur_iter)
        print("Mean metric value: {:.2f}".format(ep_metric))
        print(
            "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                len(ep_rew),
                np.mean(ep_rew),
                np.median(ep_rew),
                np.min(ep_rew),
                np.max(ep_rew),
            )
        )

    def _log_fps(self, cur_iter, log=True):
        env_steps = self.env_steps_done(cur_iter)
        end = time.time()
        self.fps = int(env_steps / (end - self.start))
        if log:
            print(
                "Updates {}, num timesteps {}, FPS {}".format(
                    cur_iter, env_steps, self.fps
                )
            )

    def env_steps_done(self, cur_iter):
        return (cur_iter + 1) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS

    def save_rewards(self, path=None):
        if not path:
            file_name = "{}_results.json".format(self.file_prefix)
            path = os.path.join(cfg.OUT_DIR, file_name)

        self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
        stats = {
            "rewards": self.mean_ep_rews,
            "fps": self.fps,
            "pos": self.mean_pos,
            "vel": self.mean_vel,
            "metric": self.mean_metric
        }
        fu.save_json(stats, path)

    def save_video(self, save_dir):
        env = make_vec_envs(
            xml_file=self.xml_file,
            training=False,
            norm_rew=False,
            save_video=True,
        )
        set_ob_rms(env, get_ob_rms(self.envs))

        env = VecVideoRecorder(
            env,
            save_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=cfg.PPO.VIDEO_LENGTH,
            file_prefix=self.file_prefix,
        )
        obs = env.reset()

        for _ in range(cfg.PPO.VIDEO_LENGTH + 1):
            _, act, _ = self.agent.act(obs)
            obs, _, _, _ = env.step(act)

        env.close()
        # remove annoying meta file created by monitor
        os.remove(
            os.path.join(save_dir, "{}_video.meta.json".format(self.file_prefix))
        )


def lr_linear_decay(optimizer, iter, total_iters, initial_lr):
    """Decreases the learning rate linearly."""
    lr = initial_lr - (initial_lr * (iter / float(total_iters)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
