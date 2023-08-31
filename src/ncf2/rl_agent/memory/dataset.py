"""Difinition of dataset for RL agent

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import NamedTuple, Union

import jax.numpy as jnp
import numpy as np
from chex import Array
from gym.spaces import Box, Dict, Discrete
from omegaconf import DictConfig


class Buffer:
    def __init__(
        self,
        observation_space: Dict,
        action_space: Union[Discrete, Box],
        env_config: DictConfig,
        model_config: DictConfig,
    ):
        """
        replay buffer

        Args:
            observation_space (Dict): observation space
            action_space (Union[Box, Discrete]): action space
            env_config (DictConfig): environment configuration
            model_config (DictConfig): model configuration
        """
        self.size = 0
        self.insert_index = 0
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_agents = env_config.num_agents
        self.model_name = model_config.name
        self.capacity = model_config.memory_size

        # dim
        if isinstance(action_space, Discrete):
            self.act_dim = []
        else:
            self.act_dim = action_space.shape

        self.obs_dim = observation_space["obs"].shape[0]
        self.num_agents = observation_space["comm"].shape[0]
        self.comm_dim = observation_space["comm"].shape[1]
        self.mask_dim = observation_space["mask"].shape[0]

        if self.model_name == "soto":
            self.comm_dim += 1  # add priority
        if self.model_name == "ncf2":
            self.obs_dim += 1  # add stop penalty
            self.comm_dim += 1  # add priority

        total_obs_dim = self.obs_dim + self.num_agents * self.comm_dim + self.mask_dim

        # buffer
        if self.model_name == "soto":
            # soto is trained with all agents in the environment simultaneously
            self.capacity = int(self.capacity / self.num_agents)
            self.observations = np.zeros(
                (self.capacity, self.num_agents, total_obs_dim),
                dtype=observation_space["obs"].dtype,
            )
            self.actions = np.zeros(
                (self.capacity, self.num_agents, *self.act_dim),
                dtype=action_space.dtype,
            )
            self.rewards = np.zeros(
                (
                    self.capacity,
                    self.num_agents,
                ),
                dtype=np.float32,
            )
            self.masks = np.zeros(
                (
                    self.capacity,
                    self.num_agents,
                ),
                dtype=np.float32,
            )
            self.next_observations = np.zeros(
                (self.capacity, self.num_agents, total_obs_dim),
                dtype=observation_space["obs"].dtype,
            )
        else:
            # other models are trained with one agent in the environment independently
            self.capacity = int(self.capacity)
            self.observations = np.zeros(
                (self.capacity, total_obs_dim), dtype=observation_space["obs"].dtype
            )
            self.actions = np.zeros(
                (self.capacity, *self.act_dim), dtype=action_space.dtype
            )
            self.rewards = np.zeros((self.capacity,), dtype=np.float32)
            self.masks = np.zeros((self.capacity,), dtype=np.float32)
            self.next_observations = np.zeros(
                (self.capacity, total_obs_dim), dtype=observation_space["obs"].dtype
            )


class ModelInput(NamedTuple):
    base_observation: Array
    communication: Array
    agent_mask: Array


class TrainBatch(NamedTuple):
    observations: ModelInput
    actions: Array
    rewards: Array
    masks: Array
    next_observations: ModelInput

    def reshape(self):
        batch_size, num_agents = self.observations.base_observation.shape[:-1]

        base_obs = self.observations.base_observation.reshape(
            batch_size * num_agents, -1
        )
        comm = self.observations.communication.reshape(
            batch_size * num_agents, num_agents, -1
        )
        agent_mask = self.observations.agent_mask.reshape(
            batch_size * num_agents, num_agents
        )
        obs = ModelInput(base_obs, comm, agent_mask)

        act = self.actions.reshape(batch_size * num_agents, -1)
        rew = self.rewards.reshape(batch_size * num_agents)
        mask = self.masks.reshape(batch_size * num_agents)

        next_base_obs = self.next_observations.base_observation.reshape(
            batch_size * num_agents, -1
        )
        next_comm = self.next_observations.communication.reshape(
            batch_size * num_agents, num_agents, -1
        )
        next_agent_mask = self.next_observations.agent_mask.reshape(
            batch_size * num_agents, num_agents
        )
        next_obs = ModelInput(next_base_obs, next_comm, next_agent_mask)

        return TrainBatch(obs, act, rew, mask, next_obs)


class TrainBatchCollection(NamedTuple):
    sub_train_batch: TrainBatch
    coop_train_batch: TrainBatch = None


class Experience(NamedTuple):
    observations: Array
    actions: Array
    rewards: Array
    dones: Array

    @classmethod
    def reset(
        self,
        num_agents: int,
        T: int,
        observations: Array,
        actions: Array,
    ):
        """reset experience (make zeros tensor)

        Args:
            num_agents (int): number of agent
            T (int): maximum episode length
            observations (Array): agent observation
            actions (Array): agent action
        """
        observations = jnp.zeros([T + 1, *observations.shape])
        actions = jnp.zeros([T + 1, *actions.shape])
        rewards = jnp.zeros([T + 1, num_agents])
        dones = jnp.ones([T + 1, num_agents])
        return self(
            observations,
            actions,
            rewards,
            dones,
        )

    def push(
        self,
        idx: int,
        observations: Array,
        actions: Array,
        rewards: Array,
        dones: Array,
    ):
        """push agent experience to certain step

        Args:
            idx (int): inserting index
            observations (Array): agent observation
            actions (Array): agent action
            rewards (Array): reward
            dones (Array): done
        """
        observations = self.observations.at[idx].set(observations)
        actions = self.actions.at[idx].set(actions)
        rewards = self.rewards.at[idx].set(rewards)
        dones = self.dones.at[idx].set(dones)
        return Experience(
            observations,
            actions,
            rewards,
            dones,
        )


class ExperienceCollection(NamedTuple):
    sub_experience: Experience
    coop_experience: Experience = None

    @classmethod
    def reset(
        self,
        num_agents: int,
        T: int,
        observations: Array,
        actions: Array,
        coop_observations: Array = None,
        coop_actions: Array = None,
    ):
        """reset experience collection (make zeros tensor)

        Args:
            num_agents (int): number of agent
            T (int): maximum episode length
            observations (Array): agent observation
            actions (Array): agent action
            coop_observations (Array): cooperative agent observation
            coop_actions (Array): cooperative agent action
        """
        sub_experience = Experience.reset(num_agents, T, observations, actions)
        if coop_observations is not None:
            coop_experience = Experience.reset(
                num_agents, T, coop_observations, coop_actions
            )
        else:
            coop_experience = None
        return self(
            sub_experience,
            coop_experience,
        )

    def push(
        self,
        idx: int,
        observations: Array,
        actions: Array,
        rewards: Array,
        dones: Array,
        coop_observations: Array = None,
        coop_actions: Array = None,
        coop_rewards: Array = None,
    ):
        """push agent experience to certain step

        Args:
            idx (int): inserting index
            observations (Array): agent observation
            actions (Array): agent action
            rewards (Array): reward
            dones (Array): done
            coop_observations (Array): cooperative agent observation
            coop_actions (Array): cooperative agent action
            coop_rewards (Array): cooperative agent reward
        """
        sub_experience = self.sub_experience.push(
            idx, observations, actions, rewards, dones
        )
        if coop_observations is not None:
            coop_experience = self.coop_experience.push(
                idx, coop_observations, coop_actions, coop_rewards, dones
            )
        else:
            coop_experience = None
        return ExperienceCollection(
            sub_experience,
            coop_experience,
        )
