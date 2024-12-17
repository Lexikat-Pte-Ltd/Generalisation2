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

  def plist_completion(self, messages: PList) -> str:
    try:
      response = self.client.messages.create(
        model=self.config.model,
        messages=cast(Any, messages.as_native()),
        max_tokens=self.config.max_tokens,
        temperature=self.config.temperature,
      )
      text_response = response.content[0].text  # type: ignore
      assert isinstance(text_response, str)

      return text_response
    except Exception as e:
      logger.error(f"Claude API request failed: {str(e)}")
      raise

  def generate_code(self, messages: PList) -> Tuple[str, str]:
    while True:
      try:
        raw_response = self.plist_completion(messages)
        processed_code = self.extract_code(raw_response)
        logger.info(f"Processed code - \n{processed_code}")
        return processed_code, raw_response
      except Exception as e:
        logger.error(f"An error while generating code occurred: {e}")
        logger.error("Retrying...")

  @staticmethod
  def extract_code(response: str) -> str:
    regex_pattern = r"```python\n([\s\S]*?)```"
    code_match = re.search(regex_pattern, response, re.DOTALL)
    assert code_match is not None
    code_string = code_match.group(1)
    assert code_string is not None
    return code_string

  def generate_list(self, messages: PList) -> Tuple[List[str], str]:
    while True:
      try:
        raw_response = self.plist_completion(messages)
        processed_list = self.extract_list(raw_response)
        return processed_list, raw_response
      except Exception as e:
        logger.error(f"An error while generating list occurred: {e}")
        logger.error("Retrying...")

  @staticmethod
  def extract_list(response: str) -> List[str]:
    start = response.index("[")
    end = response.rindex("]") + 1
    list_string = response[start:end]
    processed_list = ast.literal_eval(list_string)
    assert isinstance(processed_list, list)
    assert all(isinstance(item, str) for item in processed_list)
    return processed_list
