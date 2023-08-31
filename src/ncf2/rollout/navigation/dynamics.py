"""Definition of rollout dynamics in navigation env

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.kinematic_dynamics import (
    _build_compute_next_state,
    _build_get_relative_positions,
)
from jaxman.env.navigation.core import AgentObservation, TaskInfo, TrialInfo
from jaxman.env.navigation.dynamics import _build_inner_step
from jaxman.env.navigation.observe import _build_observe
from ncf2.rollout.utils import _build_compute_agent_intention


def _build_rollout_step(env_info: EnvInfo, agent_info: AgentInfo, actor_fn: Callable):
    _env_step = _build_inner_step(env_info, agent_info)
    _observe = _build_observe(env_info, agent_info)
    _compute_intentions = _build_compute_agent_intention(env_info, agent_info, actor_fn)

    def _step(
        state: AgentState,
        actions: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
        actor_params: FrozenDict,
    ) -> Tuple[AgentObservation, Array, Array, TrialInfo]:
        """
        compute next step based on environmental step
        In addition to the environmental step function, this step function calculates the intention and stores it in AgentObservation.

        Args:
            state (AgentState): current state
            actions (AgentAction): selected action
            task_info(TaskInfo): task information (i.e., goal status)
            trial_info (TrialInfo): trial status
            actor_params (FrozenDict): actor parameters

        Returns:
            Tuple[AgentObservation, Array, Array, TrialInfo]: next observations, rewards, dones, new_trial_info
        """
        next_state, rews, dones, new_trial_info = _env_step(
            state, actions, task_info, trial_info
        )
        next_observatinos = _observe(next_state, task_info, jnp.logical_not(dones))
        next_intentions = _compute_intentions(next_observatinos, actor_params)
        next_observatinos = next_observatinos._replace(intentions=next_intentions)

        return next_state, next_observatinos, rews, dones, new_trial_info

    return jax.jit(_step)
