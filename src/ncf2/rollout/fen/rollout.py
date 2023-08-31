"""
jax jit compiled FEN rollout function

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, FrozenDict, PRNGKey, assert_shape
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.navigation.core import AgentObservation, TaskInfo, TrialInfo
from jaxman.env.navigation.dynamics import _build_inner_step
from jaxman.env.navigation.instance import Instance
from jaxman.env.task_generator import sample_valid_start_goal
from omegaconf.dictconfig import DictConfig
from tensorflow_probability.substrates import jax as tfp

from ...memory.experience import Experience as CoopExperience
from ...memory.experience import FenExperience as SubExperience
from ..base_dynamics.utils import _build_calc_agent_obs
from .dynamics import (
    _build_rollout_step,
    _build_sample_controller_actions,
    _build_sample_sub_agent_actions,
    _calc_controller_agent_obs,
)

tfd = tfp.distributions
tfb = tfp.bijectors


class Carry(NamedTuple):
    episode_steps: int
    state: AgentState
    task_info: TaskInfo
    trial_info: TrialInfo
    agent_obs: AgentObservation
    best_actions: Array
    controller_obs: AgentObservation
    meta_act: Array
    key: PRNGKey
    ego_actor_params: FrozenDict
    ego_critic_params: FrozenDict
    coop_actor_params: FrozenDict
    controller_params: FrozenDict
    sub_experience: SubExperience
    controller_experience: CoopExperience
    env_rewards: Array
    rewards: Array
    coop_rewards: Array
    regrets: Array
    path_length: Array
    delay: Array
    num_coop: Array
    dones: Array

    @classmethod
    def reset(
        self,
        env_info: EnvInfo,
        agent_info: AgentInfo,
        init_rad: Array,
        num_sub_policy,
        sample_type: str,
        key: PRNGKey,
        single_actor_params: FrozenDict,
        ego_actor_params: FrozenDict,
        ego_critic_params: FrozenDict,
        coop_actor_params: FrozenDict,
        controller_params: FrozenDict,
        _calc_agent_obs: Callable,
        inner_rollout_fn: Callable,
    ):
        """
        reset environment

        Args:
            env_info (EnvInfo): environment basic information
            agent_info (AgentInfo): Agent Kinematic information
            init_rad (Array): Extra agent radius used during initialization
            sample_type (str): agent initial position sampling function type
            key (PRNGKey): random key
            coop_actor_params (FrozenDict): actor network parameters
            controller_params (FrozenDict): meta actor network parameters
            _observe (Callable): observation function
            _extended_observe (Callable): extended obseravtion function
            _calc_agent_obs (Callable): calculate agent observation function
            inner_rollout_fn (Callable): rollout single agent episode function

        Returns:
            Carry: carry
        """
        episode_steps = 0
        num_agents = env_info.num_agents

        key, subkey = jax.random.split(key)
        starts, start_rots, goals = sample_valid_start_goal(
            subkey,
            agent_info.rads + init_rad,
            env_info.sdf_map,
            env_info.num_agents,
            sample_type,
        )
        task_info = TaskInfo(starts=starts, start_rots=start_rots, goals=goals)

        state = AgentState(
            pos=task_info.starts,
            rot=task_info.start_rots,
            vel=jnp.zeros_like(task_info.start_rots),
            ang=jnp.zeros_like(task_info.start_rots),
        )
        trial_info = TrialInfo().reset(env_info.num_agents)

        key, agent_obs, best_actions = _calc_agent_obs(
            key,
            state,
            jnp.zeros((num_agents,), dtype=bool),
            task_info,
            single_actor_params,
        )

        env_rewards = jnp.zeros((num_agents,))
        rewards = jnp.zeros((num_agents,))
        coop_rewards = jnp.zeros((num_agents,))
        regrets = jnp.zeros((num_agents,))
        path_length = jnp.ones((num_agents,))
        num_coop = jnp.zeros((num_agents,))
        dones = jnp.array([False] * num_agents)
        coop_actions = jnp.zeros((num_agents,), dtype=int)

        agent_index = jnp.arange(num_agents)
        key, subkey = jax.random.split(key)
        rollout_key = jax.random.split(subkey, num_agents)
        delay, _, _ = jax.vmap(
            inner_rollout_fn, in_axes=[None, 0, 0, 0, 0, 0, 0, None]
        )(
            single_actor_params,
            state,
            agent_obs.observations,
            task_info,
            dones,
            agent_index,
            rollout_key,
            True,
        )
        assert_shape(delay, (num_agents,))

        # for meta policy
        controller_obs = _calc_controller_agent_obs(agent_obs, rewards, 0)

        sub_experience = SubExperience.reset(
            num_agents,
            num_sub_policy,
            env_info.timeout,
            agent_obs,
            jnp.zeros((num_agents, 2)),
        )
        controller_experience = CoopExperience.reset(
            num_agents, env_info.timeout, controller_obs, jnp.zeros((num_agents,))
        )

        return Carry(
            episode_steps,
            state,
            task_info,
            trial_info,
            agent_obs,
            best_actions,
            controller_obs,
            coop_actions,
            key,
            ego_actor_params,
            ego_critic_params,
            coop_actor_params,
            controller_params,
            sub_experience,
            controller_experience,
            env_rewards,
            rewards,
            coop_rewards,
            regrets,
            path_length,
            -delay,
            num_coop,
            dones,
        )


def build_rollout_episode(
    instance: Instance,
    env_config: DictConfig,
    model_config: DictConfig,
    action_space,
    inner_rollout_fn: Callable,
    _sample_ego_action: Callable,
    _sample_coop_action: Callable,
    _meta_actor_fn: Callable = None,
    _critic_fn: Callable = None,
    evaluate: bool = False,
) -> Callable:
    """
    build rollout multi agent episode function

    Args:
        instance (Instance): environment instance
        env_config (DictConfig): environment configuration
        model_config (DictConfig): model configuration
        inner_rollout_fn (Callable): single agent rollout function
        _sample_actions (Callable): sample rl base agent function
        _sample_meta_actions (Callable): sample meta agent function
        _critic_fn (Callable): critic function, return q value.
        evaluate (bool): Whether to include noise in the agent's actions. Defaults to False.

    Returns:
        Callable: rollout function
    """

    # define environment basic information
    env_info, agent_info, _ = instance.export_info()
    T = model_config.T
    num_sub_policy = model_config.num_sub_policy
    num_agents = env_config.num_agents
    sample_type = env_config.sample_type

    _env_step = _build_inner_step(
        env_info,
        agent_info,
    )
    _calc_agent_obs = _build_calc_agent_obs(
        env_config, model_config, instance, inner_rollout_fn
    )
    _step = _build_rollout_step(
        env_config,
        model_config,
        _env_step,
        _calc_agent_obs,
        _critic_fn,
    )
    _sample_agent_actions = _build_sample_sub_agent_actions(
        _sample_ego_action,
        _sample_coop_action,
        model_config,
        evaluate,
    )
    _sample_controller_actions = _build_sample_controller_actions(
        _meta_actor_fn,
        evaluate,
    )
    coop_act_dist = tfd.Categorical(probs=jnp.array([0.3, 0.7]))
    agent_dist = tfd.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3]))

    def _rollout_episode(
        key: PRNGKey,
        single_actor_params: FrozenDict,
        ego_actor_params: FrozenDict,
        coop_actor_params: FrozenDict,
        controller_params: FrozenDict,
        ego_critic_params: FrozenDict,
        carry: Carry = None,
    ):
        if not carry:
            carry = Carry.reset(
                env_info,
                agent_info,
                env_config.init_rad,
                num_sub_policy,
                sample_type,
                key,
                single_actor_params,
                ego_actor_params,
                ego_critic_params,
                coop_actor_params,
                controller_params,
                _calc_agent_obs,
                inner_rollout_fn,
            )

        def act_and_step(carry: Carry):
            task_info = carry.task_info
            coop_actor_params = carry.coop_actor_params
            # mask for agent who don't move in this time step
            not_finished_agent = ~carry.dones
            change_action = carry.episode_steps % T == 0
            # sample meta action
            key, controller_action, controller_probs = _sample_controller_actions(
                carry.key,
                controller_params,
                carry.controller_obs,
                carry.meta_act,
                change_action,
            )
            if evaluate:
                controller_action = controller_action
            else:
                coop_act = coop_act_dist.sample(
                    seed=key, sample_shape=(num_agents,)
                ).reshape(-1)
                coop_index = agent_dist.sample(
                    seed=key, sample_shape=(num_agents,)
                ).reshape(-1)
                controller_action = coop_index * coop_act + controller_action * (
                    1 - coop_act
                )

            # sample sub policy action
            key, array_actions, sub_policy_mask = _sample_agent_actions(
                carry.key,
                ego_actor_params,
                coop_actor_params,
                carry.agent_obs,
                controller_action,
            )
            # mask agent actions
            masked_action = jax.vmap(lambda action, action_mask: action * action_mask)(
                array_actions,
                not_finished_agent,
            )

            (
                next_state,
                next_sub_obs,
                next_controller_obs,
                env_rews,
                sub_rews,
                meta_rews,
                regs,
                regrets,
                done,
                new_trial_info,
                next_best_actions,
                key,
            ) = _step(
                carry.episode_steps,
                carry.state,
                carry.agent_obs,
                masked_action,
                carry.best_actions,
                sub_policy_mask,
                controller_probs,
                not_finished_agent,
                carry.env_rewards,
                carry.regrets,
                task_info,
                carry.trial_info,
                ego_actor_params,
                ego_critic_params,
                key,
            )

            env_rewards = carry.env_rewards + env_rews
            rewards = carry.rewards + sub_rews
            coop_rewards = carry.coop_rewards + meta_rews
            regrets = carry.regrets + regs
            path_length = carry.path_length + (~done)
            delay = carry.delay + (~done)
            num_coop = carry.num_coop + (controller_action < num_sub_policy)

            # update experience
            sub_experience = carry.sub_experience
            sub_experience = sub_experience.push(
                carry.episode_steps,
                sub_policy_mask,
                carry.agent_obs,
                array_actions,
                sub_rews,
                regs,
                done,
            )
            controller_experience = carry.controller_experience
            controller_experience = controller_experience.push(
                carry.episode_steps,
                carry.controller_obs,
                controller_action,
                meta_rews,
                regs,
                done,
            )

            carry = Carry(
                episode_steps=carry.episode_steps + 1,
                state=next_state,
                task_info=task_info,
                trial_info=new_trial_info,
                agent_obs=next_sub_obs,
                best_actions=next_best_actions,
                controller_obs=next_controller_obs,
                meta_act=controller_action,
                key=key,
                ego_actor_params=carry.ego_actor_params,
                ego_critic_params=carry.ego_critic_params,
                coop_actor_params=carry.coop_actor_params,
                controller_params=carry.controller_params,
                sub_experience=sub_experience,
                controller_experience=controller_experience,
                env_rewards=env_rewards,
                rewards=rewards,
                coop_rewards=coop_rewards,
                regrets=regrets,
                path_length=path_length,
                delay=delay,
                num_coop=num_coop,
                dones=done,
            )

            return carry

        def cond(carry):
            return ~jnp.all(carry.dones)

        carry = jax.lax.while_loop(cond, act_and_step, carry)
        return carry

    return jax.jit(_rollout_episode, static_argnames={"use_random_action"})
