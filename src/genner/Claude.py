import ast
import re
from typing import Any, List, Tuple, cast

from anthropic import Anthropic
from loguru import logger

from src.config import ClaudeConfig
from src.types import Message, PList, TaggedMessage

from .Base import Genner


class ClaudeGenner(Genner):
  def __init__(self, client: Anthropic, config: ClaudeConfig):
    self.client = client
    self.config = config

  def plist_completion(self, messages: PList) -> Tuple[bool, str]:
    try:
      response = self.client.messages.create(
        model=self.config.model,
        messages=cast(Any, messages.as_native()),
        max_tokens=self.config.max_tokens,
        temperature=self.config.temperature,
      )
      text_response = response.content[0].text  # type: ignore
      assert isinstance(text_response, str)

      return True, text_response
    except Exception as e:
      logger.error(f"Claude API request failed: {str(e)}")
      return False, ""

  def generate_code(self, messages: PList) -> Tuple[bool, str, str]:
    try:
      completion_succeed, raw_response = self.plist_completion(messages)

      if not completion_succeed:
        logger.error("Code generation failed")
        return False, "", ""

      processing_succeed, processed_code = self.extract_code(raw_response)
      if not processing_succeed:
        logger.error("Code extraction failed")
        return False, "", ""

      return True, processed_code, raw_response
    except Exception as e:
      logger.error(f"An unexpected error generating code with claude occurred: {e}")
      return False, "", ""

  @staticmethod
  def extract_code(response: str) -> Tuple[bool, str]:
    try:
      regex_pattern = r"```python\n([\s\S]*?)```"
      code_match = re.search(regex_pattern, response, re.DOTALL)
      assert code_match is not None
      code_string = code_match.group(1)
      assert code_string is not None

      return True, code_string
    except Exception as e:
      logger.error(f"An error while extracting code occurred: {e}")
      return False, ""

  def generate_list(self, messages: PList) -> Tuple[bool, List[str], str]:
    try:
      completion_succeed, raw_response = self.plist_completion(messages)
      if not completion_succeed:
        logger.error("List generation failed")
        return False, [], ""

      processing_succeed, processed_list = self.extract_list(raw_response)
      if not processing_succeed:
        logger.error("List extraction failed")
        return False, [], ""

      return True, processed_list, raw_response
    except Exception as e:
      logger.error(f"An error while generating list occurred: {e}")
      return False, [], ""

  @staticmethod
  def extract_list(response: str) -> Tuple[bool, List[str]]:
    try:
      start = response.index("[")
      end = response.rindex("]") + 1
      list_string = response[start:end]
      processed_list = ast.literal_eval(list_string)
      assert isinstance(processed_list, list)
      assert all(isinstance(item, str) for item in processed_list)

      return True, processed_list
    except Exception as e:
      logger.error(f"An unexpected error while extracting list occurred: {e}")
      return False, []
