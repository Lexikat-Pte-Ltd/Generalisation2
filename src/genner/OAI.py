import ast
import json
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

  def plist_completion(self, messages: PList) -> Tuple[bool, str]:
    try:
      response = self.client.chat.completions.create(
        model=self.config.model,
        # response_format={"type": "json_object"},
        messages=cast(Any, messages.as_native()),
        max_tokens=self.config.max_tokens,
        temperature=self.config.temperature,
      )
      assert isinstance(response.choices[0].message.content, str)

      return True, response.choices[0].message.content
    except Exception as e:
      logger.error(f"OpenAI API request failed: {str(e)}")
      return False, ""

  def generate_code(self, messages: PList) -> Tuple[List[str], str, str]:
    try:
      completion_succeed, raw_response = self.plist_completion(messages)

      if not completion_succeed:
        logger.error("Code generation failed")
        return ["Code generation failed"], "", ""

      processing_problems, processed_code = self.extract_code(raw_response)

      if len(processing_problems) > 0:
        logger.error("Code extraction failed")
        return processing_problems + ["Code extraction failed"], "", ""

      logger.info(f"Processed code - \n{processed_code}")

      return [], processed_code, raw_response
    except Exception as e:
      logger.error(f"An error while generating code occured: {e}.")

      return [f"An error while generating code occured: {e}"], "", ""

  @staticmethod
  def extract_code(response: str) -> Tuple[List[str], str]:
    try:
      # Extract code from the response
      regex_pattern = r"```python\n([\s\S]*?)```"
      code_match = re.search(regex_pattern, response, re.DOTALL)
      assert code_match is not None

      code_string = code_match.group(1)
      assert code_string is not None

      return [], code_string
    except Exception as e:
      logger.error(f"An error while extracting code occured: {e}.")
      return [f"An error while extracting code occured: {e}"], ""

  def generate_list(self, messages: PList) -> Tuple[List[str], List[str], str]:
    try:
      completion_succeed, raw_response = self.plist_completion(messages)

      if not completion_succeed:
        logger.error("List generation failed")
        return ["List generation failed"], [], ""

      processing_problems, processed_list = self.extract_list(raw_response)

      if len(processing_problems) > 0:
        logger.error("List extraction failed")
        return processing_problems + ["List extraction failed"], [], ""

      logger.info(f"Processed list - \n{processed_list}")

      return [], processed_list, raw_response
    except Exception as e:
      logger.error(f"An error while generating list occured: {e}.")
      return ["An error while generating list occured"], [], ""

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
