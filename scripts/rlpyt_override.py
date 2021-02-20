from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

import psutil
import time
import torch
import math
from collections import deque, defaultdict

import pandas as pd
import numpy as np

import cv2, os

import gym
from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper, info_to_nt
from rlpyt.envs.base import EnvSpaces, EnvStep

from rlpyt.replays.non_sequence.frame import UniformReplayFrameBuffer

try:
    import procgen
except:
    print('Failed to import Procgen env. `pip install -e .` and try again.')


class GymEnvWrapperFixed(GymEnvWrapper):
    def step(self, a):
        a = int(a.item())
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info)
        try:
            info = info.coins
        except:
            pass
        return EnvStep(obs, r, d, info)

    def seed(self, seed):
        pass


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_colors, self._height, self._width),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        obs = obs.astype(np.uint8)
        if len(obs.shape) == 4:
            obs = obs[0]
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        obs = obs.transpose(2,0,1)
        return obs

class FramePermute(gym.ObservationWrapper):
    def __init__(self, env, dim_permutation=[0,1,2]):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self.dim_permutation = dim_permutation
        old_shape = np.array(self.observation_space.shape)
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=old_shape[dim_permutation],
            dtype=np.uint8,
        )
        self.observation_space = new_space

    def observation(self, obs):
        obs = obs.transpose(*self.dim_permutation)
        return obs

def make_env(*args, info_example=None, **kwargs):
    if info_example is None:
        if 'procgen' in kwargs['id']:
            import re
            env_id, num_levels = kwargs['id'].split('.') # procgen-bigfish-v0.500 -> procgen-bigfish-v0 , 500 levels

            env_maker  = lambda args: WarpFrame(
                gym.make(id=env_id,
                        start_level=0,
                        num_levels=int(num_levels),
                        paint_vel_info=True,
                        distribution_mode='easy',
                        use_sequential_levels=True),
                width=80, height=104,grayscale=False)
            env = env_maker(None)
        return GymEnvWrapperFixed(env) # gym.make(*args, **kwargs)
    else:
        return GymEnvWrapper(EnvInfoWrapper(
            gym.make(*args, **kwargs), info_example))

def evaluate_agent(self, itr):
        self.itr = itr
        if itr > 0:
            self.pbar.stop()
        logger.log("Evaluating agent...")
        self.agent.eval_mode(itr)  # Might be agent in sampler.
        eval_time = -time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_time += time.time()
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

def _log_infos(self, traj_infos=None):
    if traj_infos is None:
        traj_infos = self._traj_infos
    if traj_infos:
        for k in traj_infos[0]:
            if not k.startswith("_"):
                logger.record_tabular_misc_stat(k,
                    [info[k] for info in traj_infos])

    if self._opt_infos:
        for k, v in self._opt_infos.items():
            try:
                logger.record_tabular_misc_stat(k, v)
            except:
                v = [x.item() for x in v]
                logger.record_tabular_misc_stat(k, v)

            self.TF_logger.log({'name':k, 'value':np.mean(v),'step':self.cum_steps})

    self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


def log_diagnostics_custom(self, itr, eval_traj_infos=None, eval_time=0):
    if not eval_traj_infos:
        logger.log("WARNING: had no complete trajectories in eval.")
    steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
    logger.record_tabular('StepsInEval', steps_in_eval)
    logger.record_tabular('TrajsInEval', len(eval_traj_infos))
    self._cum_eval_time += eval_time
    logger.record_tabular('CumEvalTime', self._cum_eval_time)

    if itr > 0:
        self.pbar.stop()
    self.save_itr_snapshot(itr)
    new_time = time.time()
    self._cum_time = new_time - self._start_time
    train_time_elapsed = new_time - self._last_time - eval_time
    new_updates = self.algo.update_counter - self._last_update_counter
    new_samples = (self.sampler.batch_size * self.world_size *
        self.log_interval_itrs)
    updates_per_second = (float('nan') if itr == 0 else
        new_updates / train_time_elapsed)
    samples_per_second = (float('nan') if itr == 0 else
        new_samples / train_time_elapsed)
    replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
        new_samples)
    cum_replay_ratio = (self.algo.batch_size * self.algo.update_counter /
        ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
    cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size

    self.cum_steps = cum_steps

    if self._eval:
        logger.record_tabular('CumTrainTime',
            self._cum_time - self._cum_eval_time)  # Already added new eval_time.
    logger.record_tabular('Iteration', itr)
    logger.record_tabular('CumTime (s)', self._cum_time)
    logger.record_tabular('CumSteps', cum_steps)
    logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
    logger.record_tabular('CumUpdates', self.algo.update_counter)
    logger.record_tabular('StepsPerSecond', samples_per_second)
    logger.record_tabular('UpdatesPerSecond', updates_per_second)
    logger.record_tabular('ReplayRatio', replay_ratio)
    logger.record_tabular('CumReplayRatio', cum_replay_ratio)
    self._log_infos(eval_traj_infos)
    logger.dump_tabular(with_prefix=False)

    self._last_time = new_time
    self._last_update_counter = self.algo.update_counter
    if itr < self.n_itr - 1:
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    ## TF log
    self.TF_logger.log({'name':'Iteration', 'value':itr,'step':cum_steps})
    self.TF_logger.log({'name':'CumTime (s)', 'value':self._cum_time,'step':cum_steps})

    self.TF_logger.log({'name':'CumSteps', 'value':cum_steps,'step':cum_steps})
    self.TF_logger.log({'name':'CumCompletedTrajs', 'value':self._cum_completed_trajs,'step':cum_steps})
    self.TF_logger.log({'name':'CumUpdates', 'value':self.algo.update_counter,'step':cum_steps})
    self.TF_logger.log({'name':'StepsPerSecond', 'value':samples_per_second,'step':cum_steps})
    self.TF_logger.log({'name':'UpdatesPerSecond', 'value':updates_per_second,'step':cum_steps})

    self.TF_logger.log({'name':'ReplayRatio', 'value':replay_ratio,'step':cum_steps})
    self.TF_logger.log({'name':'CumReplayRatio', 'value':cum_replay_ratio,'step':cum_steps})

    DL = pd.DataFrame(eval_traj_infos).to_dict('list')

    for k,v in DL.items():
        self.TF_logger.log({'name':k, 'value':np.mean(v),'step':cum_steps})