import ast
import re
from typing import Any, List, Tuple, cast

from openai import OpenAI
from loguru import logger

from src.config import OAIConfig
from src.types import PList

from .Base import Genner


class OAIGenner(Genner):
  def __init__(self, client: OpenAI, config: OAIConfig):
    super().__init__("oai")

    self.client = client
    self.config = config

  def plist_completion(self, messages: PList) -> str:
    try:
      response = self.client.chat.completions.create(
        model=self.config.model,
        # response_format={"type": "json_object"},
        messages=cast(Any, messages.as_native()),
        max_tokens=self.config.max_tokens,
        temperature=self.config.temperature,
      )
      assert isinstance(response.choices[0].message.content, str)

      return response.choices[0].message.content
    except Exception as e:
      logger.error(f"OpenAI API request failed: {str(e)}")
      raise

  def generate_code(self, messages: PList) -> Tuple[str, str]:
    while True:
      try:
        raw_response = self.plist_completion(messages)

        processed_code = self.extract_code(raw_response)

        logger.info(f"Processed code - \n{processed_code}")

        return processed_code, raw_response
      except Exception as e:
        logger.error(f"An error while generating code occured: {e}")
        logger.error("Retrying...")

  @staticmethod
  def extract_code(response: str) -> str:
    # Extract code from the response
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
        logger.error(f"An error while generating list occured: {e}")
        logger.error("Retrying...")

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
