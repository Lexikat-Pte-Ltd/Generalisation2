import json
import re
import ast
from typing import List, Tuple

from result import Result, Ok, Err

from .Ollama import OllamaGenner
from .config import QwenConfig


class QwenGenner(OllamaGenner):
    def __init__(self, config: QwenConfig):
        super().__init__(config, "qwen")

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        try:
            regex_pattern = r"```python\n([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            assert code_match is not None, "`code_match` is None"

            code_string = code_match.group(1)
            assert code_string is not None, "`code_string` is None"

            return Ok(code_string)
        except Exception as e:
            return Err(
                "QwenGenner.extract_code: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        # Try JSON parsing first
        try:
            # Remove markdown code block markers and "json" label
            json_str = response.replace("```json", "").replace("```", "").strip()

            expected_keys = ["strategies", "strats", "strategy", "Strategies", "Strats"]

            for key in expected_keys:
                if key in json_str:
                    processed_list = json.loads(json_str)[key]
                    break
            else:
                return Err(
                    "QwenGenner.extract_list: No matching strategy keys found in the response, "
                    f"`response`: \n{response}\n"
                    f"`expected_keys`: \n{expected_keys}\n"
                )

            # Validate types
            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(isinstance(item, str) for item in processed_list), (
                "All items in `processed_list` must be strings"
            )

            return Ok(processed_list)
        except Exception as e:
            pass_1_exception = e

        try:
            match = re.search(r"```(.*)```", response, re.DOTALL)

            if not match:
                return Err(
                    "QwenGenner.extract_list: No backticks (```) found in the response, "
                    f"`response`: \n{response}\n"
                )

            # Extract the content inside the backticks (group 1 of the match)
            content_inside_backticks = match.group(1).strip()

            processed_list = []
            # Split the extracted content into individual lines
            lines = content_inside_backticks.strip().split("\n")

            for line in lines:
                # Check if the line starts with a hyphen
                if line.strip().startswith("-"):
                    # Remove the leading hyphen and any surrounding whitespace
                    list_item = line.strip().lstrip("- ").strip()
                    processed_list.append(list_item)

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "QwenGenner.extract_list: Failed to parse response as JSON and also failed to extract strategies from backticks, "
                f"`pass_1_exception`: \n{pass_1_exception}\n"
                f"`pass_2_exception`: \n{e}\n"
                f"`response`: \n{response}\n"
            )
