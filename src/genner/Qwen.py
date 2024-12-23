import json
import re
import ast
from typing import List

from .Base import OllamaGenner
from src.config import QwenConfig


class QwenGenner(OllamaGenner):
  def __init__(self, config: QwenConfig):
    super().__init__(config, "qwen")

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
    # Remove markdown code block markers and "json" label
    json_str = response.replace("```json", "").replace("```", "").strip()

    # Parse the JSON string
    data = json.loads(json_str)
    if "strategies" in data:
      processed_list = data["strategies"]
    elif "strats" in data:
      processed_list = data["strats"]
    else:
      raise ValueError("No strategies found in the response")

    # Validate types
    assert isinstance(processed_list, list), "Result must be a list"
    assert all(
      isinstance(item, str) for item in processed_list
    ), "All items must be strings"

    return processed_list
