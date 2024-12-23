import json
from pathlib import Path
from typing import List

from loguru import logger
from .BaseAgent import BaseAgent
from .EnvAgent import EnvAgent
from .StrategyAgent import StrategyAgent

__all__ = ["BaseAgent", "EnvAgent", "StrategyAgent"]


def save_agent_data(
  strat_agent: StrategyAgent,
  env_agent: EnvAgent,
  chosen_strat: str,
  space_freed: float,
  save_path: Path | str,
):
  data = {
    "strat_agent": strat_agent.as_native(),
    "env_agent": env_agent.as_native(),
    "chosen_strat": chosen_strat,
    "space_freed": space_freed,
  }

  try:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.suffix != ".json":
      save_path = save_path.with_suffix(".json")

    with open(save_path, "w") as f:
      json.dump(data, f, indent=2)

    return save_path
  except Exception as e:
    logger.error(f"Failed to save agent data: {e}")
    return None


def save_list_of_agent_data(
  strat_agents: List[StrategyAgent],
  env_agent: EnvAgent,
  space_freed: float,
  save_path: Path | str,
):
  data = {
    "strat_agents": [strat_agent.as_native() for strat_agent in strat_agents],
    "env_agent": env_agent.as_native(),
    "space_freed": space_freed,
  }

  try:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.suffix != ".json":
      save_path = save_path.with_suffix(".json")

    with open(save_path, "w") as f:
      json.dump(data, f, indent=2)

    return save_path
  except Exception as e:
    logger.error(f"Failed to save agent data: {e}")
    return None
