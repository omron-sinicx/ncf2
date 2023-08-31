""" initializer of rollout functio

Author: Hikaru Asano
Affiliation: OMRON SINIC X / University of Tokyo
"""
from typing import Callable

from jaxman.env import Instance
from ncf2.rl_agent.core import Agent
from omegaconf import DictConfig

from .navigation.rollout import build_rollout_episode as build_navi_rollout
from .ncf2.rollout import build_rollout_episode as build_ncf2_rollout
from .soto.rollout import build_rollout_episode as build_soto_rollout


def _build_rollout_episode(
    instance: Instance,
    agent: Agent,
    evaluate: bool,
    model_config: DictConfig,
) -> Callable:
    """build rollout episode function

    Args:
        instance (Instance): problem instance
        agent (Agent): rl agent
        evaluate (bool): whether agent explorate or evaluate
        model_config (DictConfig): model configuration

    Returns:
        Callable: jit-compiled rollout episode function
    """
    if model_config.name == "navi":
        return build_navi_rollout(instance, agent, evaluate, model_config)
    elif model_config.name == "ncf2":
        return build_ncf2_rollout(instance, agent, evaluate, model_config)
    elif model_config.name == "soto":
        return build_soto_rollout(instance, agent, evaluate, model_config)
