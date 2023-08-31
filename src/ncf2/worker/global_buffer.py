"""Difinition of distributed woker (global buffer)

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

import threading
import time
from typing import Union

import jax
import ray
from gym.spaces import Box, Dict, Discrete
from ncf2.rl_agent.memory.dataset import (
    Buffer,
    ExperienceCollection,
    TrainBatch,
    TrainBatchCollection,
)
from ncf2.rl_agent.memory.utils import (
    _build_push_experience_to_buffer,
    _build_sample_experience,
)
from omegaconf.dictconfig import DictConfig


@ray.remote(num_cpus=1, num_gpus=0)
class GlobalBuffer:
    def __init__(
        self,
        observation_space: Dict,
        action_space: Union[Discrete, Box],
        config: DictConfig,
    ):
        """
        replay buffer

        Args:
            observations_space (Dict): observation space
            actions (Union[Discrete, Box]): action space
            config (DictConfig): configuration
        """

        self.batch_size = config.train.batch_size
        self.frame = 0
        self.batched_data = []
        self.model_name = config.coop_model.name

        self.sub_buffer = Buffer(
            observation_space, action_space, config.env, config.sub_model
        )
        self._sub_push_experience = _build_push_experience_to_buffer(
            config.sub_model.name
        )
        self._sub_sample_experience = _build_sample_experience(config.sub_model)

        if self.model_name != "navi":
            if self.model_name == "ncf2":
                action_space = Discrete(2)
            self.coop_buffer = Buffer(
                observation_space, action_space, config.env, config.coop_model
            )
            self._coop_push_experience = _build_push_experience_to_buffer(
                config.coop_model.name
            )
            self._coop_sample_experience = _build_sample_experience(config.coop_model)

        self.key = jax.random.PRNGKey(0)
        self.lock = threading.Lock()

    def num_data(self) -> int:
        """
        return buffer size. called by another ray actor

        Returns:
            int: buffer size
        """
        size = self.sub_buffer.size
        size_id = ray.put(size)
        return size_id

    def run(self):
        """
        prepare data
        """
        self.background_thread = threading.Thread(
            target=self._prepare_data, daemon=True
        )
        self.background_thread.start()

    def _prepare_data(self):
        """
        prepare batched data for training
        """
        while True:
            if len(self.batched_data) <= 10:
                data = self._sample_batch()
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_batched_data(self):
        """
        get one batch of data, called by learner.
        """

        if len(self.batched_data) == 0:
            data = self._sample_batch()
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, experience: ExperienceCollection):
        """
        add rollout episode experience to buffer

        Args:
            experience (ExperienceCollection): rollout episode experience
        """
        with self.lock:
            self._sub_push_experience(self.sub_buffer, experience.sub_experience)
            if self.model_name != "navi":
                self._coop_push_experience(self.coop_buffer, experience.coop_experience)
            del experience

    def _sample_batch(self) -> TrainBatch:
        """
        sample batched data

        Returns:
            TrainBatch: sampled batch data
        """
        with self.lock:
            self.key, sub_data = self._sub_sample_experience(
                self.key,
                self.sub_buffer,
                self.sub_buffer.num_agents,
                self.sub_buffer.comm_dim,
            )
            data = TrainBatchCollection(sub_data)
            if self.model_name != "navi":
                self.key, coop_data = self._coop_sample_experience(
                    self.key,
                    self.coop_buffer,
                    self.coop_buffer.num_agents,
                    self.coop_buffer.comm_dim,
                )
                data = data._replace(coop_train_batch=coop_data)

            self.frame += 1
            return data
