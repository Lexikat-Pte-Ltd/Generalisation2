from typing import List
from pydantic import BaseModel


class AppConfig(BaseModel):
    dev: bool
    _model_name: str
    code_host_cache_path: str
    container_ids: List[str]
    main_container_idx: int

    # These 2 below are mutually exclusive
    dynamic_container: bool
    docker_compose_dir: str

    train_data_save_folder: str

    class SpecialEGCConfig(BaseModel):
        count: int
        max_retries: int

    special_egc: SpecialEGCConfig

    class StrategyListConfig(BaseModel):
        max_retries: int

    strategy_list: StrategyListConfig

    class StrategyCodeConfig(BaseModel):
        count: int
        max_retries: int

    strategy_code: StrategyCodeConfig
