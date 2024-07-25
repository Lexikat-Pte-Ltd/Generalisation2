from __future__ import annotations
from typing import List

from pydantic import BaseModel
from textwrap import dedent

from src.types import Message, TaggedMessage


class EnvironmentInfo(BaseModel):
    # From `free -m`
    total_system_memory: float
    available_system_memory: float
    running_memory: float
    # From `df -m /`
    total_storage: float
    available_storage: float

    def diff(self, other: EnvironmentInfo):
        return EnvironmentInfo(
            # From `free -m`
            total_system_memory=self.total_system_memory - other.total_system_memory,
            available_system_memory=self.available_system_memory
            - other.available_system_memory,
            running_memory=self.running_memory - other.running_memory,
            # From `df -m /`
            total_storage=self.total_storage - other.total_storage,
            available_storage=self.available_storage - other.available_storage,
        )

    def __str__(self):
        return dedent(f"""From `free -m` :
        - total_system_memory: {self.total_system_memory}
        - available_system_memory: {self.available_system_memory}
        - running_memory: {self.running_memory}
        From `df -m /` : 
        - total_storage: {self.total_storage}
        - available_storage: {self.available_storage} """)

    def files_are_deleted(self, fresh_env_info: EnvironmentInfo):
        """Assume self as the oldest object.

        Args:
            fresh_env_info (EnvironmentInfo): Freshest environment info.

        Returns:
            bool: If files are indeed managed to be deleted
        """
        available_storage = self.available_storage - fresh_env_info.available_storage

        if available_storage >= 0:
            return False
        else:
            return True


class EnvAgentRunData(BaseModel):
    tagged_chat_history: List[TaggedMessage]
    special_env_info_getter_codes: List[str]


class CommonAgentRunData(BaseModel):
    tagged_chat_history: List[TaggedMessage]
    strat: str
    space_freed: int
