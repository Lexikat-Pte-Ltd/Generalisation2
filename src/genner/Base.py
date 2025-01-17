from abc import ABC, abstractmethod
from src.config import (
  OllamaConfig,
  DeepseekConfig,
  OAIConfig,
  WizardCoderConfig,
  QwenConfig,
  ClaudeConfig,
)
from typing import List, Tuple, Literal
from src.types import Message, TaggedMessage, PList
import httpx
from loguru import logger
from ollama import ChatResponse, chat


class Genner(ABC):
  def __init__(self, identifier: str):
    self.identifier = identifier

  @abstractmethod
  def plist_completion(self, messages: PList) -> Tuple[bool, str]:
    """Generate a single strategy based on the current chat history.

    Args:
      messages (PList): Chat history

    Returns:
      bool: Whether the completion is successful
      str: Generation result
    """
    pass

  @abstractmethod
  def generate_code(self, messages: PList) -> Tuple[List[str], str, str]:
    """Generate a single strategy based on the current chat history.

    Args:
        messages (PList): Chat history

    Returns:
        List[str]: List of problems when generating the code
        str: The generated strategy
        str: The raw response
    """
    pass

  @abstractmethod
  def generate_list(self, messages: PList) -> Tuple[List[str], List[str], str]:
    """Generate a list of strategies based on the current chat history.

    Args:
        messages (PList): Chat history

    Returns:
        List[str]: List of problems when generating the list
        List[str]: The generated strategies
        str: The raw response
    """
    pass

  @abstractmethod
  def extract_code(self, response: str) -> Tuple[List[str], str]:
    """Extract the code from the response.

    Args:
        response (str): The raw response

    Returns:
        List[str]: List of problems when extracting the code
        str: The extracted code
    """
    pass

  @abstractmethod
  def extract_list(self, response: str) -> Tuple[List[str], List[str]]:
    """Extract a list of strategies from the response.

    Args:
        response (str): The raw response

    Returns:
        List[str]: List of problems when extracting the list
        List[str]: The extracted strategies
    """
    pass


class OllamaGenner(Genner):
  def __init__(self, config: OllamaConfig, identifier: str):
    super().__init__(identifier)

    self.config = config

  def plist_completion(self, messages: PList) -> Tuple[bool, str]:
    try:
      response: ChatResponse = chat(self.config.model, messages.as_native())

      if response.message.content is None:
        logger.error(
          f"No content in the response, config: {self.config.model}, messages: {messages}"
        )
        return False, ""

      return True, response.message.content
    except Exception as e:
      logger.error(
        f"An unexpected Ollama error while generating code with {self.config.name} occured: {e}, raw response: {response}"
      )
      return False, ""

  def generate_code(
    self, messages: PList, wrap_code: bool = True
  ) -> Tuple[List[str], str, str]:
    try:
      completion_succeed, raw_response = self.plist_completion(messages)

      if not completion_succeed:
        logger.error(
          f"PList completion failed, raw response: {raw_response}, messages: {messages}"
        )
        return ["PList completion failed"], "", ""

      processing_problems, processed_code = self.extract_code(raw_response)

      if len(processing_problems) > 0:
        logger.error(
          f"Code extraction failed, raw response: {raw_response}, processed code: {processed_code}"
        )
        return processing_problems + ["Code extraction failed"], "", ""

      logger.debug(
        f"Generate code succeed, raw response: {raw_response}, processed code: {processed_code}"
      )

      return [], processed_code, raw_response
    except Exception as e:
      logger.error(
        f"An unexpected error while generating code with {self.config.name} occured: {e}, raw response: {raw_response}"
      )

      return [f"An unexpected error occured, {e}"], "", ""

  def generate_list(self, messages: PList) -> Tuple[List[str], List[str], str]:
    try:
      completion_succeed, raw_response = self.plist_completion(messages)

      if not completion_succeed:
        logger.error("PList completion failed")
        return ["PList completion failed"], [], ""

      processing_problems, processed_list = self.extract_list(raw_response)

      if len(processing_problems) > 0:
        logger.error(f"List extraction failed, raw response: {raw_response}")
        return processing_problems + ["List extraction failed"], [], ""

      logger.debug(
        f"Generate list succeed, raw response: {raw_response}, processed list: {processed_list}"
      )

      return [], processed_list, raw_response
    except Exception as e:
      logger.error(
        f"An unexpected error while generating list with {self.config.name} occured: {e}, raw response: {raw_response}"
      )

      return [f"An unexpected error occured, {e}"], [], ""

  @abstractmethod
  def extract_code(self, response: str) -> Tuple[List[str], str]:
    pass

  @abstractmethod
  def extract_list(self, response: str) -> Tuple[List[str], List[str]]:
    pass
