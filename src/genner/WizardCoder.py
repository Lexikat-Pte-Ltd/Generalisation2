import re
import ast
from typing import List

from .Base import OllamaGenner
from src.config import WizardCoderConfig


class WizardCoderGenner(OllamaGenner):
  def __init__(self, config: WizardCoderConfig):
    super().__init__(config, "wizardcoder")

  @staticmethod
  def extract_code(response: str) -> str:
    # Extract code from the response
    regex_pattern = r"```python\n([\s\S]*?)```"
    code_match = re.search(regex_pattern, response, re.DOTALL)
    assert code_match is not None

    code_string = code_match.group(1)
    assert code_string is not None

    return code_string

  @staticmethod
  def extract_list(response: str) -> List[str]:
    start = response.index("[")
    end = response.rindex("]") + 1
    list_string = response[start:end]

    # Parse the string to a Python list
    processed_list = ast.literal_eval(list_string)

    assert isinstance(processed_list, list)
    assert all(isinstance(item, str) for item in processed_list)

    return processed_list

