"""
Jax implementation for NCF2 (proposed method) rollout function

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, assert_shape
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
    agent_index = jnp.arange(num_agents)

    dummy_comm = jnp.zeros((num_agents, 1, 2))
    dummy_mask = jnp.zeros((num_agents, 1))

    _calc_regret = _build_calc_regret(num_agents, _critic_fn)
    _calc_coop_rewards = _build_coop_reward(_critic_fn, env_info, model_config)
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
        worst_actions: Array,
        coop_act: Array,
        not_finished_agent: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
        # other parameters
        actor_params: FrozenDict,
        critic_params: FrozenDict,
        regrets: Array,
        key: PRNGKey,
    ) -> Tuple[Array, Array, Array]:

        next_state, sub_rews, done, new_trial_info = _env_step(
            state, real_actions, task_info, trial_info
        )

        # compute best action without considering the existence of other agents
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

        # calculate other agents that each agent considers
        coop_reward_mask = jnp.copy(observations.masks)
        assert_shape(coop_reward_mask, (num_agents, num_agents))

        # used for ablation study
        if use_fix_prio:
            regrets = agent_index

        coop_rews = _calc_coop_rewards(
            observations,
            real_actions,
            worst_actions,
            critic_params,
            coop_act,
            regs,
            regrets,
            coop_reward_mask,
        )

        next_regrets = regrets + regs
        if use_fix_prio:
            next_regrets = agent_index
        # add priority at the last of relaitve_position and traj
        next_coop_observations = calc_coop_obs(
            next_observations,
            next_regrets,
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


def calc_relative_regrets(
    own_regrets: Array, all_regrets: Array, reward_mask: Array
) -> Array:
    """
    calculate relative regret to neighbor agent

    Args:
        own_regrets (Array): my own regret
        all_regrets (Array): all agent regret
        reward_mask (Array): reward mask to represent neighbor agent

    Returns:
        Array: relative regret to neighbor agent
    """
    rel_regrets = all_regrets - own_regrets
    rel_regrets = rel_regrets * reward_mask
    rel_regrets = jnp.clip(rel_regrets, a_min=0)
    rel_regrets /= jnp.sum(all_regrets * reward_mask) + own_regrets + 0.00001
    return rel_regrets


def penalty_weight(own_regrets: Array, all_regrets: Array, reward_mask: Array) -> Array:
    """
    calculate stop penalty weight

    Args:
        own_regrets (Array): my own regret
        all_regrets (Array): all agent regret
        reward_mask (Array): reward mask to represent neighbor agent

    Returns:
        Array: stop penalty weight
    """

    neighbor_regrets = all_regrets * reward_mask
    neighbor_regrets = jnp.sum(neighbor_regrets) + own_regrets
    return own_regrets / (neighbor_regrets + 0.00001)


def _build_coop_reward(
    critic_fn: Callable, env_config: DictConfig, model_config: DictConfig
) -> Callable:
    """
    build calculate cooperative agent reward function

    Args:
        critic_fn (Callable): critic apply function
        env_config (DictConfig): environment configuration
        model_config (DictConfig): model configuration

    Returns:
        Callable: calculate cooperative agent reward
    """
    num_agents = env_config.num_agents
    dummy_pos = jnp.zeros((num_agents, 1, 2))
    dummy_mask = jnp.zeros((num_agents, 1))

    def calc_coop_rewards(
        observations: AgentObservation,
        real_actions: Array,
        worst_actions: Array,
        critic_params: FrozenDict,
        coop_actions: Array,
        regs: Array,
        regrets: Array,
        reward_mask: Array,
    ) -> Array:
        """
        calculate cooperative agent reward

        Args:
            observations (Array): agent observation
            real_actions (Array): real action (agent taken action)
            worst_actions (Array): worst action (action when agent consider all neighbor agent message)
            critic_params (Params): critic network parameters
            coop_actions (Array): cooperative agent observation
            sub_rewards (Array): sub agent reward
            regs (Array): current step regret
            regrets (Array): sum of regret up to current step
            reward_mask (Array): reward mask to represent neighbor agent

        Returns:
            Array: cooperative agent reward
        """

        def calc_regret(regrets, reward_mask):
            reg = jnp.sum(regrets * reward_mask)
            return reg

        observations = observations._replace(
            relative_positions=dummy_pos, intentions=dummy_pos, masks=dummy_mask
        ).split_observation()
        real_q1, real_q2 = critic_fn(
            {"params": critic_params},
            observations,
            real_actions,
        )
        real_q = jnp.minimum(real_q1, real_q2)
        worst_q1, worst_q2 = critic_fn(
            {"params": critic_params},
            observations,
            worst_actions,
        )
        worst_q = jnp.minimum(worst_q1, worst_q2)

        improve = (real_q - worst_q).reshape(
            num_agents,
        )
        # Do not consider improves of agents which is stopped by coop-agents
        improve = improve * coop_actions
        improve = jnp.clip(improve, a_min=0)

        relative_regrets = jax.vmap(calc_relative_regrets, in_axes=[0, None, 0])(
            regrets, regrets, reward_mask
        )
        assert_shape(relative_regrets, (num_agents, num_agents))

        # stop penalty
        # stop penalty is calculated according to how much the q-value dropped when agent stopped
        stop_penalty = regs
        # if model_config.use_relative_penalty:
        weight = jax.vmap(penalty_weight, in_axes=[0, None, 0])(
            regrets, regrets, reward_mask
        )
        coop_rewards = stop_penalty * weight * (~coop_actions.astype(bool))
        # else:
        #     coop_rewards = stop_penalty * (~coop_actions.astype(bool))

        # add contribution if coop agent decide to stop.
        # this part evaluate how stop action contribute to other agent regret minimization
        assert_shape(reward_mask, (num_agents, num_agents))
        # if model_config.use_improve:
        # improve is weighted by relative regrets
        contributions = improve * relative_regrets
        assert_shape(contributions, (num_agents, num_agents))
        contributions = jax.vmap(calc_regret, in_axes=[0, 0])(
            contributions, reward_mask
        )
        # else:
        #     contributions = jnp.sum(relative_regrets, axis=-1)
        assert_shape(contributions, (num_agents,))
        coop_rewards += (
            contributions * (~coop_actions.astype(bool)) * model_config.coop_beta
        )
        return coop_rewards

    return jax.jit(calc_coop_rewards)


@partial(jax.jit, static_argnames={"pos_type", "traj_length"})
def calc_coop_obs(
    observations: AgentObservation,
    regrets: Array,
) -> AgentObservation:
    """
    calculate coop agent observations

    Args:
        observations (AgentObservation): sub agent observation
        regrets (Array): regrets

    Returns:
        Callable: cooperative agent observation
    """

    relative_regrets = jax.vmap(calc_relative_regrets, in_axes=[0, None, 0])(
        regrets, regrets, observations.masks
    )
    relative_regrets = jnp.expand_dims(relative_regrets, -1)

    regret_weigth = jax.vmap(penalty_weight, in_axes=[0, None, 0])(
        regrets, regrets, observations.masks
    )
    regret_weigth = regret_weigth.reshape(-1, 1)

    return observations._replace(extra_obs=regret_weigth, extra_comm=relative_regrets)
