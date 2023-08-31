"""AgentObservation for training agent network

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Tuple, Union

import jax
import numpy as np
from chex import PRNGKey
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from gym.spaces import Box, Dict, Discrete
from omegaconf import DictConfig

from .memory.dataset import TrainBatch
from .sac.sac import SAC, _update_sac_jit
from .sac.sac import build_sample_action as build_sample_sac_action
from .sac.sac import create_sac_agent, restore_sac_actor


class Agent(NamedTuple):
    greedy_agent: SAC
    sub_agents: SAC
    coop_agents: SAC = None


class AgentParams(NamedTuple):
    sub_params: FrozenDict
    critic_params: FrozenDict = None
    coop_params: FrozenDict = None


def create_agent(
    observation_space: Dict,
    action_space: Union[Box, Discrete],
    config: DictConfig,
    key: PRNGKey,
) -> Tuple[SAC, PRNGKey]:
    """create sac agent

    Args:
        observation_space (Dict): agent observation space
        action_space (Union[Box, Discrete]): agent action space
        config (DictConfig): configuraion
        key (PRNGKey): random variable key

    Returns:
        Tuple[TrainState,TrainState,TrainState,TrainState,PRNGKey]: SAC agent and key
    """
    greedy_agent, key = create_sac_agent(
        observation_space, action_space, config.greedy_model, key
    )
    sub_agent, key = create_sac_agent(
        observation_space, action_space, config.sub_model, key
    )

    if config.coop_model.name == "navi":
        return Agent(greedy_agent, sub_agent, None), key

    elif config.coop_model.name == "ncf2":
        obs_dim = observation_space["obs"].shape[0] + 1
        num_agents, comm_dim = observation_space["comm"].shape
        comm_dim += 1
        obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        comm_space = Box(
            low=-1.0, high=1.0, shape=(num_agents, comm_dim), dtype=np.float32
        )
        mask_space = Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        observation_space = Dict(
            {
                "obs": obs_space,
                "comm": comm_space,
                "mask": mask_space,
            }
        )
        action_space = Discrete(2)

    elif config.coop_model.name == "soto":
        obs_dim = observation_space["obs"].shape[0]
        num_agents, comm_dim = observation_space["comm"].shape
        comm_dim += 1
        obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        comm_space = Box(
            low=-1.0, high=1.0, shape=(num_agents, comm_dim), dtype=np.float32
        )
        mask_space = Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        observation_space = Dict(
            {
                "obs": obs_space,
                "comm": comm_space,
                "mask": mask_space,
            }
        )

    coop_agent, key = create_sac_agent(
        observation_space, action_space, config.coop_model, key
    )
    return Agent(greedy_agent, sub_agent, coop_agent), key


def build_sample_agent_action(
    actor_fn: Callable,
    is_discrete: bool,
    env_name: str,
    evaluate: bool,
    model_config: DictConfig,
):
    return build_sample_sac_action(actor_fn, is_discrete, env_name, evaluate)


@partial(
    jax.jit,
    static_argnames=(
        "is_discrete",
        "auto_temp_tuning",
        "update_target",
        "train_actor",
        "model_name",
    ),
)
def _update_jit(
    key: PRNGKey,
    agent: Agent,
    batch: TrainBatch,
    gamma: float,
    tau: float,
    target_entropy: float,
    auto_temp_tuning: bool,
    update_target: bool,
    train_actor: bool,
    model_name: str,
) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, Dict]:
    """
    update agent network

    Args:
        key (PRNGKey): random varDable key
        agent (Agent): Namedtuple of agent.
        batch (TrainBatch): Train Batch
        gamma (float): gamma. decay rate
        tau (float): tau. target critic update rate
        is_discrete (bool): whether agent action space is Discrete or not
        target_entropy (float): target entropy
        auto_temp_tuning (bool): whether to update temperature
        update_target (bool): whether to update target_critic network
        train_actor (bool): whether to update actor
        model_name (str): model name, sac or dqn

    Returns:
        Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, Dict]: new key, updated SAC agent, loss informations
    """
    if model_name == "navi":
        key, new_agent, info = _update_sac_jit(
            key,
            agent,
            batch,
            gamma,
            tau,
            False,
            target_entropy,
            auto_temp_tuning,
            update_target,
            train_actor,
            model_name,
        )
    elif model_name == "ncf2":
        key, new_agent, info = _update_sac_jit(
            key,
            agent,
            batch,
            gamma,
            tau,
            True,
            target_entropy,
            auto_temp_tuning,
            update_target,
            train_actor,
            model_name,
        )
    elif model_name == "soto":
        key, new_agent, info = _update_sac_jit(
            key,
            agent,
            batch,
            gamma,
            tau,
            False,
            target_entropy,
            auto_temp_tuning,
            update_target,
            train_actor,
            model_name,
        )
    return key, new_agent, info


def restore_agent(
    agent: Agent,
    is_discrete: bool,
    is_diff_drive: bool,
    model_config: DictConfig,
    restore_dir: str,
) -> Agent:
    if isinstance(agent, SAC):
        return restore_sac_actor(agent, is_discrete, is_diff_drive, restore_dir)
