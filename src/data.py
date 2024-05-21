from __future__ import annotations

from pydantic import BaseModel


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
