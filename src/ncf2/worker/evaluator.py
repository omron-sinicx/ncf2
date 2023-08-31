"""Definition of Evaluator

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""


import csv
import threading
import time
from typing import Tuple

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import jax
import jax.numpy as jnp
import numpy as np
import ray
from flax.training.train_state import TrainState
from jaxman.env import Instance, TrialInfo
from jaxman.env.viz.viz import render_gif
from omegaconf import DictConfig

from ..rl_agent.core import Agent, AgentParams
from ..rollout.rollout import _build_rollout_episode
from .learner import Learner


@ray.remote(num_cpus=1, num_gpus=0)
class Evaluator:
    def __init__(
        self,
        learner: Learner,
        agent: Agent,
        instance: Instance,
        model_config: DictConfig,
        seed: int = 0,
    ) -> None:
        """
        Actor. collect agent rollout data

        Args:
            learner (Learner): learner. update agent parameters
            agent (Agent): rl agent
            instance (Instance): envrironment instance
            model_config (DictConfig): model configuration
            seed (int, optional): seed. Defaults to 0
        """
        self.learner = learner
        self.agent = agent
        self.instance = instance
        self.env_name = instance.env_name
        self.model_config = model_config

        self.seed = seed

        self.average_reward = jnp.zeros((self.instance.num_agents,))
        self.trial_info = []
        self.counter = 0
        self.last_counter = 0
        self.animation = None
        self.done = False
        self._rollout_fn = _build_rollout_episode(
            instance, agent, evaluate=True, model_config=model_config
        )

    def run(self) -> None:
        """
        remote run. collect agent rollout episode
        """
        self.learning_thread = threading.Thread(target=self._rollout, daemon=True)
        self.learning_thread.start()

    def _rollout(self) -> None:
        """
        collect agent rollout episode
        """
        key = jax.random.PRNGKey(self.seed)

        params = self._update_parameters()

        while True:
            time.sleep(1)
            # rollout episode
            key, subkey = jax.random.split(key)
            carry = self._rollout_fn(subkey, params, self.instance.obs)

            if self.counter % 20 == 0:
                # update parameters
                params = self._update_parameters()

            self.average_reward += carry.rewards
            self.trial_info.append(carry.trial_info)

            # render episode trajectory
            # if self.counter % 100 == 0:
            #     steps = carry.episode_steps
            #     state_traj = carry.experience.observations[:steps, :, :5]
            #     last_state = carry.state.cat()

            #     state_traj = jnp.vstack(
            #         (state_traj, jnp.expand_dims(last_state, axis=0))
            #     )
            #     dones = carry.experience.dones[:steps, :]
            #     first_dones = jnp.zeros_like(dones[0], dtype=bool)
            #     dones = jnp.vstack((jnp.expand_dims(first_dones, 0), dones))
            #     animation = render_gif(
            #         state_traj,
            #         item_traj,
            #         goal_traj,
            #         self.instance.rads,
            #         carry.task_info,
            #         carry.trial_info,
            #         dones,
            #         self.instance.is_discrete,
            #         high_quality=False,
            #         task_type=self.env_name,
            #     )
            #     self.animation = np.expand_dims(
            #         np.transpose(np.array(animation), [0, 3, 1, 2]), 0
            #     )

            self.counter += 1

    def _update_parameters(self):
        """load actor parameters from learner"""
        params_id, _ = ray.get(self.learner.get_params.remote())
        params = ray.get(params_id)
        return params

    def stats(self, interval: int) -> Tuple[float, float, TrialInfo, np.ndarray]:
        """
        report current status of actor

        Args:
            interval (int): report interval

        Returns:
            Tuple[float, float, TrialInfo, np.ndarray]: average reward, average evaluation reward, trial info, rendering animation
        """
        num_rollouts = self.counter - self.last_counter
        print("number of eval rollout: {}".format(self.counter))
        print("eval rollout speed: {}/s".format(num_rollouts / interval))
        if num_rollouts > 0:
            print(
                "eval reward: {:.4f}".format(
                    np.mean(self.average_reward) / num_rollouts
                )
            )
            average_reward = self.average_reward.copy() / num_rollouts
            trial_info = self.trial_info.copy()
            animation = self.animation

            self.average_reward = jnp.zeros((self.instance.num_agents,))
            self.trial_info = []
            self.animation = None
        else:
            average_reward = trial_info = animation = None
        self.last_counter = self.counter
        return (
            average_reward,
            trial_info,
            animation,
        )

    def evaluate(self, eval_iters: int):
        """evaluate trained actor

        Args:
            eval_iters (int): number of evaluation episode
        """
        # update parameters
        params = self._update_parameters()
        reward = []
        solved = []
        success = []
        makespan = []
        sum_of_cost = []

        for i in range(eval_iters):
            key = jax.random.PRNGKey(i)
            carry = self._rollout_fn(key, params, self.instance.obs)
            reward.append(carry.rewards.mean())
            trial_info = carry.trial_info
            success.append(int(trial_info.is_success))
            solved.append(int(np.mean(trial_info.solved)))
            if trial_info.is_success and self.env_name == "navigation":
                makespan.append(trial_info.makespan)
                sum_of_cost.append(trial_info.sum_of_cost)

        reward_mean = bs.bootstrap(np.array(reward), stat_func=bs_stats.mean)
        reward_std = bs.bootstrap(np.array(reward), stat_func=bs_stats.std)
        success_mean = bs.bootstrap(np.array(success), stat_func=bs_stats.std)
        success_std = bs.bootstrap(np.array(success), stat_func=bs_stats.std)
        solved_mean = bs.bootstrap(np.array(solved), stat_func=bs_stats.std)
        solved_std = bs.bootstrap(np.array(solved), stat_func=bs_stats.std)
        if self.env_name == "navigation":
            makespan_mean = bs.bootstrap(np.array(makespan), stat_func=bs_stats.mean)
            makespan_std = bs.bootstrap(np.array(makespan), stat_func=bs_stats.std)
            sum_of_cost_mean = bs.bootstrap(
                np.array(sum_of_cost), stat_func=bs_stats.mean
            )
            sum_of_cost_std = bs.bootstrap(
                np.array(sum_of_cost), stat_func=bs_stats.std
            )

        f = open("eval.cvs", "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "mean", "mean_lower", "mean_upper", "std"])
        csv_writer.writerow(
            [
                "reward",
                reward_mean.value,
                reward_mean.lower_bound,
                reward_mean.upper_bound,
                reward_std.value,
            ]
        )
        csv_writer.writerow(
            [
                "solved",
                solved_mean.value,
                solved_mean.lower_bound,
                solved_mean.upper_bound,
                solved_std.value,
            ]
        )
        csv_writer.writerow(
            [
                "success",
                success_mean.value,
                success_mean.lower_bound,
                success_mean.upper_bound,
                success_std.value,
            ]
        )
        if self.env_name == "navigation":
            csv_writer.writerow(
                [
                    "makespan",
                    makespan_mean.value,
                    makespan_mean.lower_bound,
                    makespan_mean.upper_bound,
                    makespan_std.value,
                ]
            )
            csv_writer.writerow(
                [
                    "sum_of_cost",
                    sum_of_cost_mean.value,
                    sum_of_cost_mean.lower_bound,
                    sum_of_cost_mean.upper_bound,
                    sum_of_cost_std.value,
                ]
            )
        f.close()
        self.done = True

    def is_eval_done(self) -> bool:
        """returns whether the evaluation has been completed or not.

        Returns:
            bool: evluation has been completed or not
        """
        return self.done
