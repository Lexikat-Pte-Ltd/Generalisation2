import ast
import re
from typing import Any, List, Tuple, cast, Optional

from anthropic import Anthropic
from result import Result, Ok, Err, UnwrapError

from src.config import ClaudeConfig
from src.types import PList

from .Base import Genner


class ClaudeGenner(Genner):
    def __init__(self, client: Anthropic, config: ClaudeConfig):
        super().__init__("claude")
        self.client = client
        self.config = config

    def plist_completion(self, messages: PList) -> Result[str, str]:
        try:
            response = self.client.messages.create(
                model=self.config.model,
                messages=cast(Any, messages.as_native()),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            text_response = response.content[0].text  # type: ignore
            assert isinstance(text_response, str)

            return Ok(text_response)
        except Exception as e:
            return Err(
                "ClaudeGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )

    def generate_code(
        self, messages: PList
    ) -> Result[Tuple[str, str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            extracted_code = self.extract_code(raw_response).unwrap()
            return Ok((extracted_code, raw_response))
        except UnwrapError as e:
            # If raw_response is None, plist_completion failed.
            # Otherwise, extract_code failed.
            error_message = (
                "ClaudeGenner.generate_code: Unwrap error,\n"
                f"`self.config.name`: {self.config.name}\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "ClaudeGenner.generate_code: Unexpected error,\n"
                f"`self.config.name`: {self.config.name}\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        try:
            regex_pattern = r"```python\n([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            assert code_match is not None, "`code_match` is None"
            code_string = code_match.group(1)
            assert code_string is not None, "`code_string` is None"

            return Ok(code_string)
        except Exception as e:
            return Err(
                "ClaudeGenner.extract_code: Unexpected error,\n"  #
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    def generate_list(
        self, messages: PList
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap()
            processed_list = self.extract_list(raw_response).unwrap()
            return Ok((processed_list, raw_response))
        except UnwrapError as e:
            # If raw_response is None, plist_completion failed.
            # Otherwise, extract_list failed.
            error_message = (
                "ClaudeGenner.generate_list: Unwrap error,\n"
                f"`self.config.name`: {self.config.name}\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "ClaudeGenner.generate_list: Unexpected error,\n"
                f"`self.config.name`: {self.config.name}\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            list_string = response[start:end]
            processed_list = ast.literal_eval(list_string)
            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(isinstance(item, str) for item in processed_list), (
                "All item in `processed_list` is not string"
            )

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "ClaudeGenner.extract_list: Unexpected error,\n"  #
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )
