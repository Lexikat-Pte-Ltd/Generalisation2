from typing import List
from pydantic import BaseModel


class AppSettings(BaseModel):
    dev: bool
    container_ids: List[str]
    dynamic_container: bool
