from .Base import Genner
from abc import abstractmethod
from src.types import PList
from typing import Tuple

class DreamGenner(Genner):
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
