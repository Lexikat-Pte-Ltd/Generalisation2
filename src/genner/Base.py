from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from loguru import logger
from ollama import ChatResponse, Client, chat
from result import Result, Ok, Err, UnwrapError, is_err

from .config import (
    OllamaConfig,
)
from src.helper import timeout
from src.typing.message import Message



class Genner(ABC):
    def __init__(self, identifier: str):
        self.identifier = identifier

    @abstractmethod
    def plist_completion(self, messages: List[Message]) -> Result[str, str]:
        pass

    @abstractmethod
    def generate_code(
        self, messages: List[Message]
    ) -> Result[Tuple[str, str], Tuple[str, Optional[str]]]:
        pass

    @abstractmethod
    def generate_list(
        self, messages: List[Message]
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        pass

    @staticmethod
    @abstractmethod
    def extract_code(response: str) -> Result[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def extract_list(response: str) -> Result[List[str], str]:
        pass
