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
  def plist_completion(self, messages: PList) -> str:
    pass

  @abstractmethod
  def generate_code(self, messages: PList) -> Tuple[str, str]:
    pass

  @abstractmethod
  def generate_list(self, messages: PList) -> Tuple[List[str], str]:
    pass

  @abstractmethod
  def extract_code(self, response: str) -> str:
    pass

  @abstractmethod
  def extract_list(self, response: str) -> List[str]:
    pass


class OllamaGenner(Genner):
  def __init__(self, config: OllamaConfig, identifier: str):
    super().__init__(identifier)

    self.config = config

  def plist_completion(self, messages: PList) -> str:
    response: ChatResponse = chat(self.config.model, messages.as_native())
    assert response.message.content is not None
    return response.message.content
    

  # def plist_completion(self, messages: PList) -> str:
  #   payload = {
  #     "model": self.config.model,
  #     "messages": messages.as_native(),
  #     "stream": self.config.stream,
  #   }

  #   try:
  #     response = httpx.post(self.config.endpoint, json=payload)
  #     response.raise_for_status()

  #     return response.json()["message"]["content"]
  #   except httpx.HTTPStatusError as e:
  #     logger.error(f"API request failed: {str(e)}")
  #     raise

  def generate_code(self, messages: PList, wrap_code: bool = True) -> Tuple[str, str]:
    try:
      raw_response = self.plist_completion(messages)
      processed_code = self.extract_code(raw_response)

      logger.info(f"Processed code - \n{processed_code}")

      return processed_code, raw_response
    except httpx.HTTPStatusError as e:
      if e.response and e.response.status_code == 404:
        logger.error(
          f"{self.config.name} API is not available. Please check your {self.config.name} API endpoint."
        )
      else:
        logger.error(
          f"An error while generating code with {self.config.name} occured: {e}"
        )

      return "", ""
    except Exception as e:
      logger.error(
        f"An error while generating code with {self.config.name} occured: {e}"
      )
      logger.error("Retrying...")

      return "", ""

  def generate_list(self, messages: PList) -> Tuple[List[str], str]:
    try:
      raw_response = self.plist_completion(messages)
      processed_list = self.extract_list(raw_response)

      return processed_list, raw_response
    except httpx.HTTPStatusError as e:
      if e.response and e.response.status_code == 404:
        logger.error(
          f"{self.config.name} API is not available. Please check your {self.config.name} API endpoint."
        )
      else:
        logger.error(
          f"An error while generating list with {self.config.name} occured: {e}"
        )

      logger.error(f"Raw response: {raw_response}")

      return [], ""
    except Exception as e:
      logger.error(
        f"An error while generating list with {self.config.name} occured: {e}"
      )

      logger.error(f"Raw response: {raw_response}")

      return [], ""

  @abstractmethod
  def extract_code(self, response: str) -> str:
    pass

  @abstractmethod
  def extract_list(self, response: str) -> List[str]:
    pass
