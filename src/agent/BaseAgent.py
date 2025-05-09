from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger

from src.helper import (
    represent_multiline_str,
)
from src.types import PList, TaggedMessage, TaggedPList


class BaseAgent:
    def __init__(self) -> None:
        self.tagged_chat_history = TaggedPList()
        self.in_con_path: Path | str = "/"

    @property
    def chat_history(self) -> PList:
        """Use this property function if you want to get the chat history of this agent.

        Returns:
                        PList: Chat history.
        """
        return self.tagged_chat_history.as_plist()

    def debug_log_tch(self, context="Logging TCH ...") -> None:
        """Use this to log TCH (Tagged Chat History)

        Args:
                        context (str, optional): Extra log before the main log.
        """
        if context is not None:
            logger.debug(context)
        logger.debug(self.tagged_chat_history)

    def get_initial_tch(self) -> TaggedPList:
        """Get initial tagged chat history. Should be implemented by child classes.

        Returns:
                        TaggedPList: Initial tagged chat history
        """
        raise NotImplementedError

    # def save_data(self, folder: Path | str, prefix: str, extra_data: dict | None = None):
    #   """Save agent data to YAML file.

    #   Args:
    #       folder (Path | str): Folder to save data to
    #       prefix (str): Prefix for filename
    #       extra_data (dict, optional): Additional data to save. Defaults to None.
    #   """
    #   yaml.add_representer(str, represent_multiline_str)

    #   tch_len = len(self.tagged_chat_history.as_plist().messages)
    #   formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H:%M")
    #   file_name = f"{prefix}_{formatted_datetime}_run_data_at_{tch_len}.yaml"

    #   save_data = {
    #     "tag": f"[history, {prefix}]",
    #     "tagged_chat_history": self.tagged_chat_history.as_native(),
    #   }

    #   if extra_data:
    #     save_data.update(extra_data)

    #   with open(Path(folder) / file_name, "w") as yaml_file:
    #     yaml.dump(
    #       save_data,
    #       yaml_file,
    #       width=100,
    #       default_flow_style=False,
    #       sort_keys=False,
    #     )
