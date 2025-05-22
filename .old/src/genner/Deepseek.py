import ast
import re
from typing import List

from result import Result, Ok, Err

from src.config import DeepseekConfig

from .Base import OllamaGenner


class DeepseekGenner(OllamaGenner):
    def __init__(self, config: DeepseekConfig):
        super().__init__(config, "deepseek")

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        try:
            # Extract code from the response
            regex_pattern = r"```python\n([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            assert code_match is not None, "`code_match` is None"

            code_string = code_match.group(1)
            assert code_string is not None, "`code_string` is None"

            return Ok(code_string)
        except Exception as e:
            return Err(
                "DeepseekGenner.extract_code: Unexpected error,\n"  #
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            list_string = response[start:end]

            # Parse the string to a Python list
            processed_list = ast.literal_eval(list_string)

            assert isinstance(processed_list, list)
            assert all(isinstance(item, str) for item in processed_list)

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "DeepseekGenner.extract_list: Unexpected error,\n"  #
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )
