import re
import ast
from typing import List

from result import Result, Ok, Err

from .Base import OllamaGenner
from src.config import WizardCoderConfig


class WizardCoderGenner(OllamaGenner):
    def __init__(self, config: WizardCoderConfig):
        super().__init__(config, "wizardcoder")

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
                "WizardCoderGenner.extract_code: Unexpected error,\n"
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

            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(isinstance(item, str) for item in processed_list), (
                "All items in `processed_list` must be strings"
            )

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "WizardCoderGenner.extract_list: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )
