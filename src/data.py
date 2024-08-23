from __future__ import annotations
from typing import List

from pydantic import BaseModel
from textwrap import dedent

from src.types import TaggedMessage


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
        return (
            "From `free -m` :\n"
            f"- total_system_memory: {self.total_system_memory}\n"
            f"- available_system_memory: {self.available_system_memory}\n"
            f"- running_memory: {self.running_memory}\n"
            "From `df -m /` :\n"
            f"- total_storage: {self.total_storage}\n"
            f"- available_storage: {self.available_storage}"
        )

    def total_files_deleted(self, fresh_env_info: EnvironmentInfo):
        """Assume self as the oldest object.

        Args:
            fresh_env_info (EnvironmentInfo): Freshest environment info.

        Returns:
            float: Difference in storage
            bool: If files are indeed managed to be deleted
        """
        old_available_storage = self.available_storage
        new_available_storage = fresh_env_info.available_storage

        storage_diff = old_available_storage - new_available_storage

        if storage_diff > 0:
            return storage_diff, True
        else:
            return storage_diff, False
