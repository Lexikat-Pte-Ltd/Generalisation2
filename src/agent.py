from typing import Dict, List

from result import Result

from src.genner.Base import Genner
from src.prompts import get_sp_egc_req_prompt, get_system_prompt


def generate_special_environment_getter_code(
    genner: Genner,
    env_infos: List[str],
    special_env_infos: List[str],
) -> Result[str, str]:
    messages = [
        {
            "role": "system",
            "content": get_system_prompt(),
        },
        {
            "role": "user",
            "content": get_sp_egc_req_prompt(
                basic_env_infos=env_infos, special_env_infos=special_env_infos
            ),
        },
    ]

    return genner.generate_code(messages)
