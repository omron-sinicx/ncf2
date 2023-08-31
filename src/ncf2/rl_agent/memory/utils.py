"""utility functions for memory

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from ncf2.utils import split_obs_and_comm
from omegaconf.dictconfig import DictConfig

from .dataset import Buffer, Experience, TrainBatch


def _build_push_experience_to_buffer(model_name: str) -> Callable:
    def _push_experience_to_buffer(buffer: Buffer, experience: Experience):
        idx = buffer.insert_index
        num_agents = experience.rewards.shape[-1]

        for i in range(num_agents):
            last_idx = np.where(experience.dones[:, i])[0][0]
            if idx + last_idx + 1 > buffer.capacity:
                extra_length = (idx + last_idx + 1) - buffer.capacity
                last_idx = last_idx - extra_length
                buffer.size = buffer.capacity

            buffer.observations[idx : idx + last_idx + 1] = np.copy(
                experience.observations[: last_idx + 1, i]
            )
            buffer.actions[idx : idx + last_idx + 1] = np.copy(
                experience.actions[: last_idx + 1, i]
            )
            buffer.rewards[idx : idx + last_idx + 1] = np.copy(
                experience.rewards[: last_idx + 1, i]
            )
            buffer.masks[idx : idx + last_idx + 1] = np.copy(
                1 - experience.dones[: last_idx + 1, i]
            )
            buffer.next_observations[idx : idx + last_idx + 1] = np.copy(
                experience.observations[1 : last_idx + 2, i]
            )
            idx += last_idx + 1

            buffer.size = min(buffer.size + last_idx + 1, buffer.capacity)
            buffer.insert_index = idx % buffer.capacity

    def _push_soto_experience_to_buffer(buffer: Buffer, experience: Experience):
        idx = buffer.insert_index
        num_agents = experience.rewards.shape[-1]
        last_idx = 0

        for i in range(num_agents):
            temp_last_idx = np.where(experience.dones[:, i])[0][0]
            last_idx = np.max([last_idx, temp_last_idx])

        if idx + last_idx + 1 > buffer.capacity:
            extra_length = (idx + last_idx + 1) - buffer.capacity
            last_idx = last_idx - extra_length
            buffer.size = buffer.capacity

        buffer.observations[idx : idx + last_idx + 1] = np.copy(
            experience.observations[: last_idx + 1, :]
        )
        buffer.actions[idx : idx + last_idx + 1] = np.copy(
            experience.actions[: last_idx + 1, :]
        )
        buffer.rewards[idx : idx + last_idx + 1] = np.copy(
            experience.rewards[: last_idx + 1, :]
        )
        buffer.masks[idx : idx + last_idx + 1] = np.copy(
            1 - experience.dones[: last_idx + 1, :]
        )
        buffer.next_observations[idx : idx + last_idx + 1] = np.copy(
            experience.observations[1 : last_idx + 2, :]
        )
        idx += last_idx + 1

        buffer.size = min(buffer.size + last_idx + 1, buffer.capacity)
        buffer.insert_index = idx % buffer.capacity

    if model_name == "soto":
        return _push_soto_experience_to_buffer
    else:
        return _push_experience_to_buffer


def _build_sample_experience(
    model_config: DictConfig,
):
    batch_size = model_config.batch_size

    def _sample_experience(
        key: PRNGKey,
        buffer: Buffer,
        num_agents: int,
        comm_dim: int,
    ):
        index = np.random.randint(buffer.size, size=batch_size)

        all_obs = buffer.observations[index]
        agent_obs = split_obs_and_comm(all_obs, num_agents, comm_dim)
        acts = buffer.actions[index]
        rews = buffer.rewards[index]
        masks = buffer.masks[index]
        next_all_obs = buffer.next_observations[index]
        next_agent_obs = split_obs_and_comm(next_all_obs, num_agents, comm_dim)

        data = TrainBatch(agent_obs, acts, rews, masks, next_agent_obs)

        return key, data

    return _sample_experience
