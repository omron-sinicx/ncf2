"""Utility functions for rollout functions

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.kinematic_dynamics import (
    _build_compute_next_state,
    _build_get_relative_positions,
)
from jaxman.env.navigation.core import AgentObservation, TaskInfo
from jaxman.env.navigation.dynamics import _build_is_solved
from jaxman.env.utils import get_scans
from jaxman.planner.dwa import create_planner
from ncf2.rl_agent.core import build_sample_agent_action
from omegaconf import DictConfig


def _build_compute_agent_intention(
    env_info: EnvInfo, agent_info: AgentInfo, actor_fn: Callable
) -> Callable:
    num_agents = env_info.num_agents
    _compute_next_state = _build_compute_next_state(env_info)
    _compute_relative_pos = _build_get_relative_positions(env_info)

    comm_dim = 4  # (rel_pos, next_rel_pos) * 2

    dummy_comm = jnp.zeros((num_agents, 1, comm_dim))
    dummy_mask = jnp.zeros((num_agents, 1))

    def _compute_agents_intentions(
        observations: AgentObservation,
        actor_params: FrozenDict,
    ) -> Array:
        """
        compute agent intentions. intention represent where agent will move

        Args:
            observations (AgentObservation): current agents' observations
            actor_params (FrozenDict): actor parameters

        Returns:
            Array: intentions
        """
        state = observations.state
        observations = observations.split_observation()
        dummy_observation = observations._replace(
            communication=dummy_comm, agent_mask=dummy_mask
        )
        actions, _ = actor_fn({"params": actor_params}, dummy_observation)
        next_possible_state = jax.vmap(_compute_next_state)(state, actions, agent_info)
        intentions = _compute_relative_pos(state, next_possible_state)
        return intentions

    return jax.jit(_compute_agents_intentions)


def _build_calc_regret(
    num_agents: int,
    _critic_fn: Callable,
):

    dummy_comm = jnp.zeros((num_agents, 1, 4))  # (rel_pos, next_rel_pos)
    dummy_mask = jnp.zeros((num_agents, 1))

    def calc_regret(
        critic_params: FrozenDict,
        observations: AgentObservation,
        best_actions: Array,
        real_actions: Array,
        not_finished_agent: Array,
    ) -> Array:
        observations = observations.split_observation()
        dummy_observations = observations._replace(
            communication=dummy_comm, agent_mask=dummy_mask
        )
        best_q1, best_q2 = _critic_fn(
            {"params": critic_params},
            dummy_observations,
            best_actions,
        )
        best_q = jnp.minimum(best_q1, best_q2)
        real_q1, real_q2 = _critic_fn(
            {"params": critic_params},
            dummy_observations,
            real_actions,
        )
        real_q = jnp.minimum(real_q1, real_q2)
        regs = (best_q - real_q).reshape(-1)
        regs = regs * not_finished_agent
        regs = jnp.clip(regs, a_min=0)
        return regs

    return jax.jit(calc_regret)


def build_individual_rollout_steps(
    env_info: EnvInfo,
    agent_info: AgentInfo,
    actor_fn: Callable,
    model_config: DictConfig,
):
    class Carry(NamedTuple):
        t: int
        key: PRNGKey
        actor_params: FrozenDict
        state: AgentState
        task_info: TaskInfo
        steps: Array
        last_done: Array

    num_agents = env_info.num_agents
    key = jax.random.PRNGKey(0)
    _compute_next_state = _build_compute_next_state(env_info)
    _individual_observe = _build_individual_observe(env_info, agent_info)
    is_solved = _build_is_solved(agent_info, env_info.is_discrete)
    _sample_actions = build_sample_agent_action(
        actor_fn, env_info.is_discrete, env_info.env_name, True, model_config
    )

    def _individual_stpes(carry: Carry):
        t = carry.t + 1
        obs = _individual_observe(carry.state, carry.task_info)
        key, actions = _sample_actions(carry.actor_params, obs, carry.key)
        next_state = jax.vmap(_compute_next_state)(carry.state, actions, agent_info)

        done = is_solved(next_state.pos, carry.task_info.goals)
        done = jnp.logical_or(done, carry.last_done)
        time_step_done = env_info.timeout < t
        done = jnp.logical_or(time_step_done, done)

        steps = carry.steps + jnp.logical_not(done)
        return carry._replace(
            t=t, key=key, state=next_state, steps=steps, last_done=done
        )

    def cond(carry: Carry):
        return jnp.any(carry.last_done == False)

    def _compute_steps(
        state: AgentState, task_info: TaskInfo, actor_params: FrozenDict
    ):
        steps = jnp.zeros((env_info.num_agents,))
        last_done = jnp.zeros((env_info.num_agents), dtype=bool)
        carry = Carry(0, key, actor_params, state, task_info, steps, last_done)
        carry = jax.lax.while_loop(cond, _individual_stpes, carry)
        return carry.steps

    return jax.jit(_compute_steps)


def _build_individual_observe(env_info: EnvInfo, agent_info: AgentInfo) -> Callable:
    """build individual observe function. this observe function ignore the existence of other agent

    Args:
        env_info (EnvInfo): environment base information
        agent_info (AgentInfo): agent kinematics information

    Returns:
        Callable: jit-compiled observe function
    """
    num_agents = env_info.num_agents
    num_scans = env_info.num_scans
    scan_range = env_info.scan_range
    _planner = create_planner(env_info, agent_info, is_individual_plannning=True)

    def _individual_observe(state: AgentState, task_info: TaskInfo) -> AgentObservation:
        scans = jax.vmap(get_scans, in_axes=(0, 0, None, None, None))(
            state.pos, state.rot, task_info.obs.edges, num_scans, scan_range
        )
        planner_act = _planner._act(state, task_info.goals, task_info.obs.occupancy)
        dummy_pos = jnp.zeros((num_agents, 1, 2))
        dummy_masks = jnp.zeros((num_agents, 1))
        return AgentObservation(
            state=state,
            goals=task_info.goals,
            scans=scans,
            planner_act=planner_act,
            relative_positions=dummy_pos,
            intentions=dummy_pos,
            masks=dummy_masks,
        )

    return jax.jit(_individual_observe)
