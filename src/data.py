from __future__ import annotations
from typing import Any, Dict, List

from pydantic import BaseModel
from textwrap import dedent

from src.types import TaggedMessage


class EnvironmentInfo(BaseModel):
    container_id: str
    # From `free -m`
    total_system_memory: float
    available_system_memory: float
    running_memory: float
    # From `df -m /`
    total_storage: float
    available_storage: float

    def diff(self, other: EnvironmentInfo):
        return {
            "container_id": self.container_id,
            "total_system_memory": self.total_system_memory - other.total_system_memory,
            "available_system_memory": self.available_system_memory
            - other.available_system_memory,
            "running_memory": self.running_memory - other.running_memory,
            "total_storage": self.total_storage - other.total_storage,
            "available_storage": self.available_storage - other.available_storage,
        }

    def __str__(self):
        return (
            f"Container d: {self.container_id}\n"
            "From `free -m` :\n"
            f"- total_system_memory: {self.total_system_memory}\n"
            f"- available_system_memory: {self.available_system_memory}\n"
            f"- running_memory: {self.running_memory}\n"
            "From `df -m /` :\n"
            f"- total_storage: {self.total_storage}\n"
            f"- available_storage: {self.available_storage}"
        )

    def get_total_storage_deleted(self, fresh_env_info: EnvironmentInfo):
        """Assume self as the oldest object.

        Args:
            fresh_env_info (EnvironmentInfo): Freshest environment info.

        Returns:
            float: Difference in storage
            bool: If files are indeed managed to be deleted
        """
        old_available_storage = self.available_storage
        new_available_storage = fresh_env_info.available_storage

        space_freed_diff = new_available_storage - old_available_storage  # Corrected calculation

        if space_freed_diff > 0:
            return space_freed_diff, True  # Space was freed
        else:
            # This will return (0.0, False) if no change,
            # or (negative_value, False) if space was consumed.
            return space_freed_diff, False


    def as_native(self) -> Dict[str, Any]:
        return {
            "total_system_memory": self.total_system_memory,
            "available_system_memory": self.available_system_memory,
            "running_memory": self.running_memory,
            "total_storage": self.total_storage,
            "available_storage": self.available_storage,
        }
