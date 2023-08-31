"""jax jit-compiled rollout function for navigation environment

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jaxman.env.core import AgentInfo, AgentState, EnvInfo
from jaxman.env.navigation.core import AgentObservation, TaskInfo, TrialInfo
from jaxman.env.navigation.instance import Instance
from jaxman.env.navigation.observe import _build_observe
from jaxman.env.obstacle import ObstacleMap
from jaxman.env.task_generator import sample_valid_start_goal
from ncf2.rl_agent.core import Agent, AgentParams, build_sample_agent_action
from ncf2.rl_agent.memory.dataset import ExperienceCollection
from omegaconf import DictConfig
from tensorflow_probability.substrates import jax as tfp

from ..utils import build_individual_rollout_steps
from .dynamics import _build_compute_agent_intention, _build_rollout_step

tfd = tfp.distributions
tfb = tfp.bijectors


class Carry(NamedTuple):
    episode_steps: int
    state: AgentState
    task_info: TaskInfo
    trial_info: TrialInfo
    observations: AgentObservation
    coop_observations: AgentObservation
    key: PRNGKey
    experience: ExperienceCollection
    rewards: Array
    coop_rewards: Array
    regrets: Array
    path_length: Array
    delay: Array
    dones: Array

    @classmethod
    def reset(
        self,
        env_info: EnvInfo,
        agent_info: AgentInfo,
        obs: ObstacleMap,
        key: PRNGKey,
        greedy_params: FrozenDict,
        params: AgentParams,
        _observe: Callable,
        _compute_intentions: Callable,
        _compute_individual_steps: Callable,
    ):
        episode_steps = 0
        subkey, key = jax.random.split(key)

        num_agents = env_info.num_agents
        starts, start_rots, goals = sample_valid_start_goal(
            subkey,
            agent_info.rads,
            obs,
            num_agents,
            env_info.is_discrete,
        )
        task_info = TaskInfo(starts, start_rots, goals, obs)
        state = AgentState(
            pos=task_info.starts,
            rot=task_info.start_rots,
            vel=jnp.zeros_like(task_info.start_rots),
            ang=jnp.zeros_like(task_info.start_rots),
        )
        observations = _observe(state, task_info, jnp.ones((num_agents,), dtype=bool))
        trial_info = TrialInfo().reset(env_info.num_agents)
        rewards = jnp.zeros((num_agents,))
        coop_rewards = jnp.zeros((num_agents,))
        regrets = jnp.zeros((num_agents,))
        path_length = jnp.zeros((num_agents,))
        delays = _compute_individual_steps(state, task_info, greedy_params) * (-1)
        dones = jnp.array([False] * num_agents)

        intentions = _compute_intentions(observations, params.sub_params)
        observations = observations._replace(intentions=intentions)

        priority = jnp.zeros((num_agents, num_agents, 1), dtype=float)
        weight = jnp.zeros((num_agents, 1), dtype=float)
        coop_observations = observations._replace(extra_obs=weight, extra_comm=priority)

        # initialize experience
        if env_info.is_discrete:
            actions = jnp.zeros((num_agents,))
        else:
            actions = jnp.zeros((num_agents, 2))
        coop_actions = jnp.zeros((num_agents,))
        experience = ExperienceCollection.reset(
            num_agents,
            env_info.timeout,
            observations.cat(),
            actions,
            coop_observations.cat(),
            coop_actions,
        )

        return self(
            episode_steps,
            state,
            task_info,
            trial_info,
            observations,
            coop_observations,
            key,
            experience,
            rewards,
            coop_rewards,
            regrets,
            path_length,
            delays,
            dones,
        )


def build_rollout_episode(
    instance: Instance,
    agent: Agent,
    evaluate: bool,
    model_config: DictConfig,
) -> Callable:
    """build rollout episode function

    Args:
        instance (Instance): problem instance
        actor_fn (Callable): actor function
        evaluate (bool): whether agent explorate or evaluate
        model_config (DictConfig): model configuration file

    Returns:
        Callable: jit-compiled rollout episode function
    """
    env_info, agent_info, _ = instance.export_info()

    greedy_params = agent.greedy_agent.actor.params

    _step = _build_rollout_step(
        env_info,
        agent_info,
        model_config,
        agent.sub_agents.actor.apply_fn,
        agent.sub_agents.critic.apply_fn,
    )
    _observe = _build_observe(env_info, agent_info)
    _compute_intentions = _build_compute_agent_intention(
        env_info, agent_info, agent.sub_agents.actor.apply_fn
    )
    _sample_actions = build_sample_agent_action(
        agent.sub_agents.actor.apply_fn,
        instance.is_discrete,
        instance.env_name,
        evaluate,
        model_config,
    )
    _sample_coop_actions = build_sample_agent_action(
        agent.coop_agents.actor.apply_fn,
        True,
        instance.env_name,
        evaluate,
        model_config,
    )
    _compute_individual_steps = build_individual_rollout_steps(
        env_info, agent_info, agent.sub_agents.actor.apply_fn, model_config
    )
    coop_action_dist = tfd.Categorical(probs=jnp.array([0.05, 0.95]))

    def _rollout_episode(
        key: PRNGKey,
        params: AgentParams,
        obs: ObstacleMap,
        random_action: bool = False,
        carry: Carry = None,
    ):
        if not carry:
            carry = Carry.reset(
                env_info,
                agent_info,
                obs,
                key,
                greedy_params,
                params,
                _observe,
                _compute_intentions,
                _compute_individual_steps,
            )

        def _act_and_step(carry: Carry):
            not_finished_agent = ~carry.dones
            state = carry.state

            # sample worst actions (consider all neighboring agent intentions)
            key, worst_actions = _sample_actions(
                params.sub_params, carry.observations, carry.key
            )
            # sample cooperative agent actions
            coop_agent_obs = carry.coop_observations._replace(planner_act=worst_actions)
            if random_action:
                key, subkey = jax.random.split(key)
                coop_actions = coop_action_dist.sample(
                    seed=subkey, sample_shape=(env_info.num_agents,)
                )
            else:
                key, coop_actions = _sample_coop_actions(
                    params.coop_params, coop_agent_obs, carry.key
                )
            # sample real actions
            movable_agent = not_finished_agent * coop_actions
            masks = jax.vmap(
                lambda mask, is_movable: mask * is_movable, in_axes=[0, None]
            )(carry.observations.masks, movable_agent)
            new_obs = carry.observations._replace(masks=masks)
            key, real_actions = _sample_actions(params.sub_params, new_obs, carry.key)
            # update state
            state = state._replace(
                vel=state.vel * movable_agent.reshape(-1, 1),
                ang=state.ang * movable_agent.reshape(-1, 1),
            )

            real_actions = jax.vmap(lambda action, mask: action * mask)(
                real_actions, movable_agent
            )
            (
                next_state,
                next_observations,
                next_coop_observations,
                rews,
                coop_rews,
                regs,
                dones,
                new_trial_info,
                key,
            ) = _step(
                state,
                carry.observations,
                real_actions,
                worst_actions,
                coop_actions,
                not_finished_agent,
                carry.task_info,
                carry.trial_info,
                params.sub_params,
                params.critic_params,
                carry.regrets,
                key,
            )

            rewards = carry.rewards + rews
            coop_rewards = carry.coop_rewards + coop_rews
            regrets = carry.regrets + regs
            path_length = carry.path_length + (~dones)
            delay = carry.delay + (~dones)
            experience = carry.experience.push(
                carry.episode_steps,
                carry.observations.cat(),
                real_actions,
                rews,
                dones,
                coop_agent_obs.cat(),
                coop_actions,
                coop_rews,
            )

            carry = Carry(
                episode_steps=carry.episode_steps + 1,
                state=next_state,
                task_info=carry.task_info,
                trial_info=new_trial_info,
                observations=next_observations,
                coop_observations=next_coop_observations,
                key=key,
                experience=experience,
                rewards=rewards,
                coop_rewards=coop_rewards,
                regrets=regrets,
                path_length=path_length,
                delay=delay,
                dones=dones,
            )
            return carry

        def cond(carry: Carry):
            return jnp.logical_not(jnp.all(carry.dones))

        carry = jax.lax.while_loop(cond, _act_and_step, carry)
        return carry

    return jax.jit(_rollout_episode, static_argnames={"random_action"})
