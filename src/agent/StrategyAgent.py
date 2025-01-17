import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from loguru import logger

from src.data import EnvironmentInfo
from src.genner import Genner
from src.prep import (
  GET_BASIC_ENV_PLIST_TAG,
  GET_SPECIAL_ENV_PLIST_TAG,
  GET_STRATS_REQ_PLIST_TAG,
  GET_SYSTEM_PLIST_TAG,
  get_basic_env_plist,
  get_special_env_plist,
  get_strats_req_plist,
  get_strats_regen_plist,
  get_system_plist,
)
from src.types import Message, TaggedMessage, TaggedPList
from .BaseAgent import BaseAgent


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

    # For saving purposes
    self.chosen_strat: str = ""
    self.space_freed: float = 0

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

    return local_tch

  def gen_strats(
    self,
    genner: Genner,
    model_name: str,
    max_attempts: int = 3,
  ) -> Tuple[TaggedPList, List[str], str]:
    """Generate strategies based on the current chat history.

    Logic:
    1. Generate strategies based on self.tagged_chat_history and new fresh tagged chat history
    2. If the strategies are not valid, try again
    3. If the strategies are valid, return them

    Args:
        genner (Genner): The generator to use for strategy generation
        model_name (str): The name of the model to use for strategy generation
        max_attempts (int, optional): The maximum number of attempts to make. Defaults to 3.

    Raises:
        ValueError: If the tags of the self tagged chat history do not make sense to what is intended

    Returns:
        TaggedPList: The new local tagged chat history
        TaggedPList: The new perfect tagged chat history
        List[str]: The generated strategies
        str: The raw response
    """
    local_tch = TaggedPList(
      get_strats_req_plist(
        in_con_path=self.in_con_path,
        prev_strats=self.prev_strats,
        model_name=model_name,
      )
    )

    expected_tags = [
      GET_SYSTEM_PLIST_TAG,
      GET_BASIC_ENV_PLIST_TAG,
      GET_SPECIAL_ENV_PLIST_TAG,
      f"{GET_STRATS_REQ_PLIST_TAG}({model_name})",
    ]

    is_current_tags_makes_sense = (
      self.tagged_chat_history.get_tags() + local_tch.get_tags() == expected_tags
    )

    if not is_current_tags_makes_sense:
      raise ValueError(
        f"Current tags do not make sense, tags are {self.tagged_chat_history.get_tags()} expected {expected_tags}"
      )

    for attempt in range(max_attempts):
      list_of_problems, processed_list, raw_response = genner.generate_list(
        self.tagged_chat_history.as_plist() + local_tch.as_plist(),
      )

      if len(list_of_problems) > 0:
        logger.error(
          f"List generation failed, tag history: {self.tagged_chat_history.get_tags() + local_tch.get_tags()}, retrying..."
        )

        local_tch.messages.append(
          TaggedMessage(
            message=Message(role="assistant", content=raw_response),
            tag="genned_strats(failed)",
          )
        )
        local_tch.messages.extend(
          get_strats_regen_plist(
            list_of_problems=list_of_problems,
          ),
        )
        continue

      local_tch.messages.append(
        TaggedMessage(
          message=Message(role="assistant", content=raw_response),
          tag="genned_strats(success)",
        )
      )

      return local_tch, processed_list, raw_response

    if len(list_of_problems) > 0:
      sampled_strats = random.sample(self.prev_strats, 10)
      sampled_strats_as_json = json.dumps({"strategies": sampled_strats})
      sampled_strats_as_json_formatted = f"```json\n{sampled_strats_as_json}\n```"

      local_tch.messages.append(
        TaggedMessage(
          message=Message(role="assistant", content=sampled_strats_as_json_formatted),
          tag="genned_strats(success)",
        )
      )
      return local_tch, sampled_strats, sampled_strats_as_json_formatted

    return local_tch, [], ""

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

      if basic_env_info_tag.startswith(tag):
        self.tagged_chat_history.modify_message_at_index(
          index=i, new_tagged_message=get_basic_env_plist(fresh_bs_eih)[0]
        )
        changes += 1
      elif special_env_info_tag.startswith(tag):
        self.tagged_chat_history.modify_message_at_index(
          index=i, new_tagged_message=get_special_env_plist(fresh_sp_eih)[0]
        )
        changes += 1

    return changes

  def as_native(self, omit: List[str] = []) -> Dict[str, Any]:
    native_data = {
      "chosen_strat": self.chosen_strat,
      "strats": self.strats,
      "in_con_path": str(self.in_con_path),
      "init_bs_env_info_history": [
        env_info.as_native() for env_info in self.init_bs_env_info_history
      ],
      "init_sp_env_info_history": self.init_sp_env_info_history,
      "tagged_chat_history": self.tagged_chat_history.as_native(),
      "prev_strats": self.prev_strats,
      "space_freed": self.space_freed,
    }

    for omit_key in omit:
      if omit_key in native_data:
        del native_data[omit_key]

    return native_data

  # def save_data(self, folder: Path | str, space_freed: float, strat: str):
  #   """Save strategy agent data to file.

  #   Args:
  #       folder (Path | str): Folder to save to
  #       space_freed (float): Amount of space freed
  #       strat (str): Strategy used
  #   """
  #   extra_data = {
  #     "strat": strat,
  #     "space_freed": space_freed,
  #   }

  #   # Generate strategy-specific identifier for the filename
  #   strat_identifier = "".join(
  #     [word[0].lower() for word in strat.split(" ") if word.isalpha()][:10]
  #   )

  #   super().save_data(folder, f"ca_{strat_identifier}", extra_data)
