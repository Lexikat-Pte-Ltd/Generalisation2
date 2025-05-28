from typing import List
from pydantic import BaseModel


class AppConfig(BaseModel):
    dev: bool
    _model_name: str
    host_cache_path: str
    container_ids: List[str]
    main_container_idx: int

    # These 2 belows are mutually exclusive
    dynamic_container: bool
    docker_compose_dir: str

    class SpecialEGCConfig(BaseModel):
        code_host_cache_folder: str
        count: int
        max_retries: int

    special_egc: SpecialEGCConfig
