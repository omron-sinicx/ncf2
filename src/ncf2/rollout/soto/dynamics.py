"""
Jax implementation for NCF2 (proposed method) rollout function

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey

# from ...rl_agent.core import AgentObservation
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.navigation.core import AgentObservation, TaskInfo, TrialInfo
from jaxman.env.navigation.dynamics import _build_inner_step
from jaxman.env.navigation.observe import _build_observe
from omegaconf import DictConfig

from ..utils import _build_calc_regret, _build_compute_agent_intention


def _build_rollout_step(
    env_info: EnvInfo,
    agent_info: AgentInfo,
    model_config: DictConfig,
    _actor_fn: Callable,
    _critic_fn: Callable,
) -> Callable:
    """
    build ncf2 rollout step function

    Args:
        env_config (DictConfig): environment configuration
        model_config (DictConfig): model configuratino
        _env_step (Callable): mapp env step
        _calc_agent_obs (Callable): calculate agent observation
        _critic_fn (Callable): critic apply function

    Returns:
        Callable: ncf2 rollout step function
    """
    num_agents = env_info.num_agents
    use_fix_prio = model_config.use_fix_prio
    use_regret_priority = model_config.use_regret_priority
    agent_index = jnp.arange(num_agents)

    dummy_comm = jnp.zeros((num_agents, 1, 2))
    dummy_mask = jnp.zeros((num_agents, 1))

    _calc_regret = _build_calc_regret(num_agents, _critic_fn)
    # _calc_coop_rewards = _build_coop_reward(_critic_fn, env_info, model_config)
    _env_step = _build_inner_step(env_info, agent_info)
    _observe = _build_observe(env_info, agent_info)
    _compute_intentions = _build_compute_agent_intention(
        env_info, agent_info, _actor_fn
    )

    def _coop_step(
        # current state
        state: AgentState,
        observations: AgentObservation,
        real_actions: Array,
        not_finished_agent: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
        # other parameters
        actor_params: FrozenDict,
        critic_params: FrozenDict,
        reward: Array,
        regrets: Array,
        key: PRNGKey,
    ) -> Tuple[Array, Array, Array]:

        next_state, sub_rews, done, new_trial_info = _env_step(
            state, real_actions, task_info, trial_info
        )

        best_obs = observations._replace(
            relative_positions=dummy_comm, intentions=dummy_comm, masks=dummy_mask
        ).split_observation()
        best_actions, _ = _actor_fn(
            {"params": actor_params},
            best_obs,
        )

        regs = _calc_regret(
            critic_params,
            observations,
            best_actions,
            real_actions,
            not_finished_agent,
        )

        # compute agent observation
        next_observations = _observe(next_state, task_info, jnp.logical_not(done))
        next_intentions = _compute_intentions(next_observations, actor_params)
        next_observations = next_observations._replace(intentions=next_intentions)

        coop_rews = jnp.copy(sub_rews)

        next_rewards = reward + sub_rews
        next_regrets = regrets + regs
        if use_fix_prio:
            priority = agent_index
        else:
            if use_regret_priority:
                priority = -next_regrets
            else:
                priority = next_rewards
        next_coop_observations = calc_coop_obs(
            next_observations,
            priority,
        )

        return (
            next_state,
            next_observations,
            next_coop_observations,
            sub_rews,
            coop_rews,
            regs,
            done,
            new_trial_info,
            key,
        )

    return jax.jit(_coop_step, static_argnames={"use_random_action"})


@jax.jit
def calc_fair_indicator(
    own_regrets: Array, all_regrets: Array, reward_mask: Array
) -> Array:
    """
    calculate fairness indicator

    Args:
        own_regrets (Array): my own regret
        all_regrets (Array): all agent regret
        reward_mask (Array): reward mask to represent neighbor agent

    Returns:
        Array: fairness indicator
    """
    rel_regrets = all_regrets - own_regrets
    rel_regrets = rel_regrets * reward_mask
    return rel_regrets


@jax.jit
def calc_coop_obs(
    sub_agent_obs: AgentObservation,
    priority: Array,
):
    """
    calculate meta agent observations

    Args:
        sub_agent_obs (RLAgentObservatino): sub agent observation
        priority (Array): priority
    """
    fair_indicator = jax.vmap(calc_fair_indicator, in_axes=[0, None, 0])(
        priority, priority, sub_agent_obs.masks
    )
    fair_indicator = jnp.expand_dims(fair_indicator, axis=-1)

    return sub_agent_obs._replace(extra_comm=fair_indicator)


@jax.jit
def apply_is_greedy(sub_action: Array, coop_action: Array, is_greedy: Array) -> Array:
    """
    apply greedy mask. return sub agent action if greedy

    Args:
        sub_action (Array): sub agent action
        coop_action (Array): cooperative agent action
        is_greedy (Array): is agent greedy

    Returns:
        Array: agent action.
    """
    action = sub_action * is_greedy + coop_action * (~is_greedy)
    return action
