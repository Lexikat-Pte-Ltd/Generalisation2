import json
import re
import ast
from typing import List, Tuple

from loguru import logger

from .Base import OllamaGenner
from src.config import QwenConfig


class QwenGenner(OllamaGenner):
  def __init__(self, config: QwenConfig):
    super().__init__(config, "qwen")

  @staticmethod
  def extract_code(response: str) -> Tuple[List[str], str]:
    # Extract code from the response
    try:
      regex_pattern = r"```python\n([\s\S]*?)```"
      code_match = re.search(regex_pattern, response, re.DOTALL)
      assert code_match is not None, "No code match found in the response"

      code_string = code_match.group(1)
      assert code_string is not None, "No code group number 1 found in the response"

      return [], code_string
    except Exception as e:
      logger.error(
        f"An unexpected error while extracting code occurred: {str(e)}, raw response: {response}"
      )
      return [f"An unexpected error occured, {e}"], ""

  @staticmethod
  def extract_list(response: str) -> Tuple[List[str], List[str]]:
    try:
      # Remove markdown code block markers and "json" label
      json_str = response.replace("```json", "").replace("```", "").strip()

      expected_keys = ["strategies", "strats", "strategy", "Strategies", "Strats"]

      for key in expected_keys:
        if key in json_str:
          processed_list = json.loads(json_str)[key]
          break
      else:
        return [
          f"No strategies found in the response, expected keys: {expected_keys} "
        ], []

      # Validate types
      assert isinstance(processed_list, list), "Result must be a list"
      assert all(
        isinstance(item, str) for item in processed_list
      ), "All items must be strings"
    except (AssertionError, ValueError) as e:
      logger.error(
        f"An unexpected error while extracting list occurred: {e}, raw response: {response}"
      )
      return [f"An unexpected error occured, {e}"], []

    return [], processed_list
