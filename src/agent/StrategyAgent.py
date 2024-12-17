from pathlib import Path
from typing import List, Tuple

from src.data import EnvironmentInfo
from src.genner import Genner
from src.prep import (
  get_basic_env_plist,
  get_special_env_plist,
  get_strat_req_plist,
  get_system_plist,
)
from src.types import TaggedMessage, TaggedPList
from src.agent.BaseAgent import BaseAgent


class StrategyAgent(BaseAgent):
  """Strategy Agent for generating and managing strategies"""

  def __init__(
    self,
    init_bs_env_info_history: List[EnvironmentInfo],
    init_sp_env_info_history: List[List[str]],
    prev_strats: List[str],
    in_con_path: str | Path,
  ):
    """Initialize a strategy agent for strategies generation and space clearing code generation.

    Args:
        init_bs_env_info_history (List[EnvironmentInfo]): Initial basic environment info history
        init_sp_env_info_history (List[List[str]]): Initial special environment info history
        prev_strats (List[str]): Previous strategies
        in_con_path (str | Path): In container working path
    """
    super().__init__()
    self.strats: List[str] = []
    self.in_con_path = in_con_path
    self.init_bs_env_info_history = init_bs_env_info_history
    self.init_sp_env_info_history = init_sp_env_info_history
    self.prev_strats = prev_strats

  def get_initial_tch(self) -> TaggedPList:
    local_tch = TaggedPList()

    # Add system prompt
    local_tch.messages.extend(get_system_plist(in_con_path=self.in_con_path))
    # Add basic env inclusion prompt
    local_tch.messages.extend(get_basic_env_plist(bs_eih=self.init_bs_env_info_history))
    # Add special env inclusion prompt
    local_tch.messages.extend(
      get_special_env_plist(sp_eih=self.init_sp_env_info_history)
    )
    # Add strategy gen prompt
    local_tch.messages.extend(
      get_strat_req_plist(in_con_path=self.in_con_path, prev_strats=self.prev_strats)
    )

    return local_tch

  def gen_strats(self, genner: Genner) -> Tuple[List[str], str]:
    """Generate strategies based on the current chat history.

    Args:
        genner (Genner): The generator to use for strategy generation

    Returns:
        Tuple[List[str], str]: List of strategies and their generation metadata
    """
    return genner.generate_list(self.tagged_chat_history.as_plist())

  def update_env_info_state(
    self,
    fresh_bs_eih: List[EnvironmentInfo],
    fresh_sp_eih: List[List[str]],
    basic_env_info_tag="get_basic_env_plist",
    special_env_info_tag="get_special_env_plist",
  ) -> int:
    """Update environment info state with fresh data.

    Args:
        fresh_bs_eih (List[EnvironmentInfo]): Fresh basic environment info history
        fresh_sp_eih (List[List[str]]): Fresh special environment info history
        basic_env_info_tag (str, optional): Tag for basic env info. Defaults to "get_basic_env_plist"
        special_env_info_tag (str, optional): Tag for special env info. Defaults to "get_special_env_plist"

    Returns:
        int: Number of changes made
    """
    changes = 0

    for i in range(len(self.tagged_chat_history)):
      tag = self.tagged_chat_history.messages[i].tag

      if tag == basic_env_info_tag:
        self.tagged_chat_history.modify_message_at_index(
          index=i, new_tagged_message=get_basic_env_plist(fresh_bs_eih)[0]
        )
        changes += 1
      elif tag == special_env_info_tag:
        self.tagged_chat_history.modify_message_at_index(
          index=i, new_tagged_message=get_special_env_plist(fresh_sp_eih)[0]
        )
        changes += 1

    return changes

  def save_data(self, folder: Path | str, space_freed: float, strat: str):
    """Save strategy agent data to file.

    Args:
        folder (Path | str): Folder to save to
        space_freed (float): Amount of space freed
        strat (str): Strategy used
    """
    extra_data = {
      "strat": strat,
      "space_freed": space_freed,
    }

    # Generate strategy-specific identifier for the filename
    strat_identifier = "".join(
      [word[0].lower() for word in strat.split(" ") if word.isalpha()][:10]
    )

    super().save_data(folder, f"ca_{strat_identifier}", extra_data)
