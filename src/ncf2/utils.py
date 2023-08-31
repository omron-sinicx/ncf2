"""
utility functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
"""
from typing import List, Tuple
"""

import jax
import jax.numpy as jnp
from chex import Array
from ncf2.rl_agent.memory.dataset import ModelInput


@jax.jit
def value_at_i(value: Array, i: int):
    return value[i]


@jax.jit
def compute_distance(a: Array, b: Array):
    return jnp.sqrt(jnp.sum((a - b) ** 2))


def split_obs_and_comm(
    observations: Array,
    num_agents: int,
    comm_dim: int,
) -> ModelInput:
    """split observation into agent basic observations and communications

    Args:
        observations (Array): observations, contrain basic obs and comm
        num_agents (int): number of agent in whole environment
        comm_dim (int): communication dimensions

    Returns:
        ModelInput: shaped agent basic observations and communications
    """

    batch_shape = observations.shape[:-1]
    total_comm_dim = num_agents * comm_dim
    mask_dim = num_agents
    obs = observations[..., : -total_comm_dim - mask_dim]
    comm = observations[..., -total_comm_dim - mask_dim : -mask_dim].reshape(
        [*batch_shape, num_agents, comm_dim]
    )
    mask = observations[..., -mask_dim:]
    return ModelInput(obs, comm, mask)
