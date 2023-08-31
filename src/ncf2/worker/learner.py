"""
Difinition of distributed worker (learner)

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import threading
from typing import Tuple

import jax
import numpy as np
import ray
from flax.training import checkpoints
from omegaconf.dictconfig import DictConfig

from ..rl_agent.core import Agent, AgentParams, _update_jit
from .global_buffer import GlobalBuffer


@ray.remote(num_cpus=1, num_gpus=0)
class Learner:
    def __init__(
        self,
        buffer: GlobalBuffer,
        agent: Agent,
        action_scale: np.ndarray,
        action_bias: np.ndarray,
        target_entropy: bool,
        config: DictConfig,
    ) -> None:
        """
        remote Learner

        Args:
            buffer (GlobalBuffer): global buffer to sample batch
            agent (Agent): Namedtuple contain agent components.
            action_scale (np.ndarray): action scale
            action_bias (np.ndarray): action bias
            target_entropy (bool): target value of entropy of agent actions
            config (DictConfig): configuration
        """
        self.seed = config.seed
        self.horizon = int(config.train.horizon)
        self.save_dir = f"./agent/seed{self.seed}"

        # agent
        self.buffer = buffer
        self.agent = agent

        self.agent_name = config.coop_model.name
        self.is_discrete = config.env.is_discrete
        self.num_agent = config.env.num_agents
        self.batch_size = config.train.batch_size
        self.save_interval = int(config.train.save_interval)
        self.warmup_iters = config.train.warmup_iters
        self.use_pretrained_model = config.train.use_pretrained_model

        self.action_scale = action_scale
        self.action_bias = action_bias
        self.gamma = config.train.gamma
        self.tau = config.train.tau
        self.auto_temp_tuning = config.train.auto_temp_tuning
        self.target_entropy = target_entropy

        self.done = False
        self.train_actor = (False) or self.use_pretrained_model
        self.counter = 0
        self.last_counter = 0
        self.loss = []

        self.params = AgentParams(
            agent.sub_agents.actor.params,
            agent.sub_agents.critic.params,
        )
        if self.agent.coop_agents is not None:
            self.params = self.params._replace(
                coop_params=agent.coop_agents.actor.params
            )
        self.params_id = ray.put(self.params)
        self.train_actor_id = ray.put(self.train_actor)

    def run(self) -> None:
        """
        remote run. update agent parameters
        """

        self.learning_thread = threading.Thread(target=self._train, daemon=True)
        self.learning_thread.start()

    def _train(self) -> None:
        """
        update agent parameters
        """
        key = jax.random.PRNGKey(self.seed)
        for i in range(self.horizon + 1):
            loss_info = {}
            self.train_actor = (i > self.warmup_iters) or (self.use_pretrained_model)
            data = ray.get(ray.get(self.buffer.get_batched_data.remote()))
            # time.sleep(0.03)
            (key, sub_agent, loss_info,) = _update_jit(
                key,
                self.agent.sub_agents,
                data.sub_train_batch,
                self.gamma,
                self.tau,
                self.target_entropy,
                self.auto_temp_tuning,
                True,  # carry.step % carry.target_update_period == 0,
                self.train_actor,
                "navi",
            )
            if self.agent_name != "navi":
                (key, coop_agent, loss_info,) = _update_jit(
                    key,
                    self.agent.coop_agents,
                    data.coop_train_batch,
                    self.gamma,
                    self.tau,
                    self.target_entropy,
                    self.auto_temp_tuning,
                    True,  # carry.step % carry.target_update_period == 0,
                    self.train_actor,
                    self.agent_name,
                )
            else:
                coop_agent = None
            self.agent = self.agent._replace(
                sub_agents=sub_agent, coop_agents=coop_agent
            )

            self.loss.append(loss_info)

            # store new actor params in shared memory
            if i % 3 == 0:
                self.store_params()
            if i % self.save_interval == 0:
                self.save_model()
            self.counter += 1
        self.done = True

    def store_params(self) -> None:
        """
        store up to data actor parameters in shared memory
        """
        self.params = AgentParams(
            self.agent.sub_agents.actor.params,
            self.agent.sub_agents.critic.params,
        )
        if self.agent.coop_agents is not None:
            self.params = self.params._replace(
                coop_params=self.agent.coop_agents.actor.params
            )
        self.params_id = ray.put(self.params)
        self.train_actor_id = ray.put(self.train_actor)

    def get_params(self):
        """
        return agent parameter id
        """
        return self.params_id, self.train_actor_id

    def stats(self, interval: int) -> Tuple[bool, dict, int]:
        """
        report current status of learner

        Args:
            interval (int): report interval

        Returns:
            Tuple (bool, dict, int): whether to finish training, loss dict, total update num
        """
        print("number of updates: {}".format(self.counter))
        print(
            "update speed: {}/s".format((self.counter - self.last_counter) / interval)
        )
        self.last_counter = self.counter

        loss_info = self.loss.copy()
        self.loss = []
        return self.done, loss_info

    def save_model(self):
        """
        save agent model
        """
        checkpoints.save_checkpoint(
            ckpt_dir=self.save_dir,
            target=self.agent.sub_agents.actor,
            prefix="actor_checkpoint_",
            step=self.counter,
            overwrite=True,
        )
        if self.agent.coop_agents is not None:
            checkpoints.save_checkpoint(
                ckpt_dir=self.save_dir,
                target=self.agent.coop_agents.actor,
                prefix="coop_actor_checkpoint_",
                step=self.counter,
                overwrite=True,
            )
        # checkpoints.save_checkpoint(
        #     ckpt_dir=self.save_dir,
        #     target=self.critic,
        #     prefix="critic_checkpoint_",
        #     step=self.counter,
        #     overwrite=True,
        # )
        # checkpoints.save_checkpoint(
        #     ckpt_dir=self.save_dir,
        #     target=self.target_critic,
        #     prefix="target_critic_checkpoint_",
        #     step=self.counter,
        #     overwrite=True,
        # )
        # checkpoints.save_checkpoint(
        #     ckpt_dir=self.save_dir,
        #     target=self.temp,
        #     prefix="temp_checkpoint_",
        #     step=self.counter,
        #     overwrite=True,
        # )
