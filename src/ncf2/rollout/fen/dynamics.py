""" dynamics function for FEN rollout

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from chex import Array, FrozenDict, PRNGKey
from flax import linen as fnn
from jaxman.env.core import AgentAction, AgentState
from jaxman.env.navigation.core import AgentObservation, TaskInfo, TrialInfo
from omegaconf.dictconfig import DictConfig
from tensorflow_probability.substrates import jax as tfp

from ..utils import _build_calc_regret

tfd = tfp.distributions
tfb = tfp.bijectors


@jax.jit
def calc_sub_reward(
    env_reward: Array,
    sub_policy_mask: Array,
    controller_probs: Array,
) -> Array:
    """
    calculate sub agent reward

    Args:
        env_reward (Array): environment original reward
        controller_actions (Array): controller action
        controller_probs (Array): controller action probability

    Returns:
        Array: sub agent reward
    """
    # is_ego_agent = controller_actions == 0
    # if meta agent select ego agent, sub agent reward is env rewrad
    # sub_reward = env_reward * is_ego_agent
    # if meta agent select cooperative agent, sub agent reward is sub agent select probability
    # sub_reward += controller_probs * (~is_ego_agent)
    sub_reward = jnp.sum(controller_probs[:, :3] * sub_policy_mask, axis=-1)
    return sub_reward


@jax.jit
def calc_controller_reward(
    my_reward, all_rewards: Array, t: int, pos_mask: Array
) -> Array:
    """
    calculate meta agent reward

    Args:
        my_reward (Array): own environment reward
        all_rewards (Array): all agent environment reward
        t (int): episode steps
        pos_mask (Array): relative agent position mask: (num_agents, )

    Returns:
        Array: controller reward
    """
    u_bar = my_reward / (t + 1)
    avg_u = (
        (jnp.sum(all_rewards * pos_mask) + my_reward)
        / (t + 1)
        / (jnp.sum(pos_mask) + 1)
    )
    meta_reward = u_bar / (0.1 + jnp.abs(u_bar / (avg_u + 0.0000001) - 1))
    inf_mask = jnp.isinf(meta_reward) | jnp.isnan(meta_reward)
    dummy_meta_reward = u_bar / 0.0001
    meta_reward = meta_reward * (~inf_mask) + dummy_meta_reward * inf_mask
    # meta_reward = jnp.clip(meta_reward, a_min=-25, a_max=25)
    return meta_reward


@jax.jit
def _calc_controller_obs(
    observations: Array, my_reward: Array, all_rewards: Array, t: int, pos_mask: Array
) -> Array:
    """
    calculate next meta agent observation

    Args:
        observations (Array): observation
        my_reward (Array): own environment reward
        all_rewards (Array): all agent environment reward
        t (int): episode time step
        pos_mask (Array): next relative agent mask. shape: (num_agents, num_agents)

    Returns:
        Array: controller observation
    """

    utili = all_rewards / (t + 1)
    u_bar = jnp.mean(utili)
    utili = my_reward / (t + 1)
    u_bar = (
        (jnp.sum(all_rewards * pos_mask) + my_reward)
        / (t + 1)
        / (jnp.sum(pos_mask) + 1)
    )
    rat = (utili - u_bar) / (u_bar + 0.0000001)
    inf_mask = jnp.isinf(rat) | jnp.isnan(rat)
    dummy_rat = (utili - u_bar) / 0.0001
    rat = rat * (~inf_mask) + dummy_rat * inf_mask
    # rat = jnp.clip(rat, a_min=-25, a_max=25)
    return jnp.concatenate((observations, jnp.array([rat]), jnp.array([utili])))


def _calc_controller_agent_obs(
    agent_obs: AgentObservation, env_reward: Array, t: int
) -> AgentObservation:
    """
    calculate

    Args:
        agent_obs (AgentObservation): sub agent observation
        env_reward (Array): environment reward
        t (int): episode time steps

    Returns:
        AgentObservation: controller observation
    """
    obs = jnp.copy(agent_obs.base_)
    obs = jax.vmap(_calc_controller_obs, in_axes=(0, 0, None, None, 0))(
        obs,
        env_reward,
        env_reward,
        t,
        pos_mask,
    )
    return AgentObservation(obs, relative_pos, pos_mask, traj, traj_mask)


def _build_rollout_step(
    env_config: DictConfig,
    model_config: DictConfig,
    _env_step: Callable,
    _calc_agent_obs: Callable,
    _critic_fn: Callable,
):

    num_agents = env_config.num_agents

    _calc_regret = _build_calc_regret(num_agents, model_config, _critic_fn)

    def _step(
        t,
        state: AgentState,
        agent_obs: AgentObservation,
        real_actions: Array,
        best_actions: Array,
        controller_actions: Array,
        controller_probs: Array,
        not_finished_agent: Array,
        env_rewards: Array,
        regrets: Array,
        task_info: TaskInfo,
        trial_info: TrialInfo,
        ego_actor_params: FrozenDict,
        ego_critic_params: FrozenDict,
        key: PRNGKey,
    ) -> Tuple[AgentState, AgentObservation, Array, Array, TrialInfo, Array, Array]:

        actions = AgentAction.from_array(real_actions)
        next_original_obs, env_rews, done, new_trial_info = _env_step(
            state, actions, task_info, trial_info
        )
        regs = _calc_regret(
            ego_critic_params,
            agent_obs.observations,
            best_actions,
            real_actions,
            not_finished_agent,
        )
        next_env_rewards = env_rewards + env_rews
        next_regrets = regrets + regs

        sub_rews = calc_sub_reward(env_rews, controller_actions, controller_probs)
        meta_rews = jax.vmap(calc_controller_reward, in_axes=[0, None, None, 0])(
            next_env_rewards, next_env_rewards, t, agent_obs.pos_mask
        )

        next_state = next_original_obs.state
        key, next_agent_obs, best_actions = _calc_agent_obs(
            key, next_state, done, task_info, ego_actor_params
        )
        next_controller_obs = _calc_controller_agent_obs(
            next_agent_obs,
            next_env_rewards,
            t + 1,
        )

        return (
            next_state,
            next_agent_obs,
            next_controller_obs,
            env_rews,
            sub_rews,
            meta_rews,
            regs,
            next_regrets,
            done,
            new_trial_info,
            best_actions,
            key,
        )

    return jax.jit(_step)


def _build_sample_sub_agent_actions(
    _sample_ego_action: Callable,
    _sample_coop_action: Callable,
    model_config: DictConfig,
    evaluate: bool,
):
    num_sub_policy = model_config.num_sub_policy
    agent_dist = tfd.Categorical(probs=jnp.array([1 / 3, 1 / 3, 1 / 3]))

    @jax.jit
    def _sample_one_agent_actions(
        key: PRNGKey,
        ego_actor_params: FrozenDict,
        actor_params: FrozenDict,
        agent_obs: AgentObservation,
        controller_actions: Array,
    ):
        observations = jnp.expand_dims(agent_obs.observations, axis=0)
        relative_pos = jnp.expand_dims(agent_obs.relative_pos, axis=0)
        pos_mask = jnp.expand_dims(agent_obs.pos_mask, axis=0)
        traj = jnp.expand_dims(agent_obs.traj, axis=0)
        traj_mask = jnp.expand_dims(agent_obs.traj_mask, axis=0)
        use_coop_action = controller_actions < num_sub_policy

        # sample coop sub policy action
        sub_policy_mask = (
            jnp.zeros((num_sub_policy,)).at[controller_actions].set(1)
        ) * use_coop_action
        batched_sub_policy_mask = jnp.expand_dims(sub_policy_mask, axis=0)

        key, ego_array_action, ego_greedy_action = _sample_ego_action(
            key, ego_actor_params, observations, relative_pos, pos_mask, traj, traj_mask
        )
        key, coop_array_action, coop_greedy_action = _sample_coop_action(
            key,
            actor_params,
            observations,
            relative_pos,
            pos_mask,
            traj,
            traj_mask,
            batched_sub_policy_mask,
        )

        if evaluate:
            array_action = (
                use_coop_action * coop_greedy_action
                + ~use_coop_action * ego_greedy_action
            )
        else:
            array_action = (
                use_coop_action * coop_array_action
                + ~use_coop_action * ego_array_action
            )
        dummy_agent_index = agent_dist.sample(seed=key, sample_shape=(1,)).reshape(-1)
        dummy_sub_policy_mask = (
            jnp.zeros((num_sub_policy,)).at[dummy_agent_index].set(1)
        )
        # sub_policy_mask = sub_policy_mask * use_coop_action + dummy_sub_policy_mask * (~use_coop_action)

        return array_action.reshape(-1), sub_policy_mask

    def _sample_all_agent_actions(
        key: PRNGKey,
        ego_actor_params: FrozenDict,
        actor_params: FrozenDict,
        agent_obs: AgentObservation,
        controller_actions: Array,
    ):
        key, subkey = jax.random.split(key)
        array_actions, sub_policy_masks = jax.vmap(
            _sample_one_agent_actions, in_axes=[None, None, None, 0, 0]
        )(
            subkey,
            ego_actor_params,
            actor_params,
            agent_obs,
            controller_actions,
        )
        return key, array_actions, sub_policy_masks

    return jax.jit(_sample_all_agent_actions)


def _build_sample_controller_actions(_meta_agent_apply_fn: Callable, evaluate: bool):
    # def _prob_at_action(probs, action):
    #     return probs[action]

    def _sample_meta_agent_actions(
        key: PRNGKey,
        actor_params: FrozenDict,
        agent_obs: AgentObservation,
        last_action: Array,
        change_action: bool,
    ) -> Tuple[PRNGKey, Array]:

        observations = agent_obs.observations
        relative_pos = agent_obs.relative_pos
        pos_mask = agent_obs.pos_mask
        traj = agent_obs.traj
        traj_mask = agent_obs.traj_mask
        dist = _meta_agent_apply_fn(
            {"params": actor_params},
            observations,
            relative_pos,
            pos_mask,
            traj,
            traj_mask,
        )
        probs = dist.probs

        key, subkey = jax.random.split(key)
        if evaluate:
            array_actions = jnp.argmax(probs, axis=-1)
        else:
            array_actions = dist.sample(seed=subkey)
        array_actions = change_action * array_actions + ~change_action * last_action

        # probs = jax.vmap(_prob_at_action)(probs, array_actions)
        return key, array_actions, probs

    return jax.jit(_sample_meta_agent_actions)


def _build_sample_coop_actions(
    actor_apply_fn: Callable,
    action_space,
) -> Callable:
    """
    build and compile action sampling function

    Args:
        actor_apply_fn (Callable): actor network
        action_scale (Array): action scale to align with the action bound of the environment
        action_bias (Array): action bias to align with the action bound of the environment
        model_name (str): model name. currently independent sac and communicatable sac is implemented

    Returns:
            Callable: compiled action sampling function
    """
    action_scale = jnp.array((action_space.high - action_space.low) / 2.0)
    action_bias = jnp.array((action_space.high + action_space.low) / 2.0)

    def _sample_actions(
        rng: PRNGKey,
        actor_params: FrozenDict,
        observations: Array,
        agent_pos: Array,
        pos_mask: Array,
        traj: Array,
        traj_mask: Array,
        sub_policy_mask: Array,
    ) -> Array:
        """
        sample agent action. this function called only when interacting with environment

        Args:
            rng (PRNGKey): random key
            actor_params (FrozenDict): agent actor network parameters
            observations (Array): agent observations
            agent_pos (Array): agent position
            pos_mask (Array): one hot bool representation of agent mask. one for agent own index, and 0 for other agent index. shape: (batch_size, num_agents)
            traj (Array): rollout trajectory
            traj_mask (Array): mask for rollout trajectory
            sub_policy_mask (Array): mask to represent which sub policy is selected

        Returns:
            Array: sampled agent action
        """
        rng, key = jax.random.split(rng)

        # batch_size = observations.shape[0]
        dist = actor_apply_fn(
            {"params": actor_params},
            observations,
            agent_pos,
            pos_mask,
            traj,
            traj_mask,
            sub_policy_mask,
        )

        x_t = dist.sample(seed=key)
        y_t = fnn.tanh(x_t)

        # residual network
        planner_act = observations[:, -2:]
        actions = y_t + planner_act
        actions = actions * action_scale + action_bias

        mean = dist.mean()
        mean = fnn.tanh(mean) + planner_act
        mean = mean * action_scale + action_bias
        return rng, actions, mean

    return jax.jit(_sample_actions)
