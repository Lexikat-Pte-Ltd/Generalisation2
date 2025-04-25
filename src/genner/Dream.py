import ast
import re
from typing import Any, List, Tuple, cast
from loguru import logger
from src.config import DreamConfig
from src.types import PList
import requests

from .Base import Genner


class DreamGenner(Genner):
    def __init__(self, config: DreamConfig):
        self.cfg = config

    def plist_completion(self, messages: PList) -> Tuple[bool, str]:
        url = f"{self.cfg.base_url.rstrip('/')}/generate"
        messages = cast(Any, messages.as_native())
        payload = {
            "messages": messages,
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "steps": self.cfg.steps,
            "alg": self.cfg.alg,
            "timeout": self.cfg.timeout,
        }

        try:
            r = requests.post(
                url,
                json=payload,
                timeout=self.cfg.timeout,
            )

            r.raise_for_status()

            json_payload = cast(dict[str, Any], r.json())
            text_response = cast(str, json_payload["result"])

            return True, text_response
        except requests.exceptions.HTTPError as e:
            logger.error(
                "Dream API request failed, "
                f"`e.response.status_code`: {e.response.status_code}\n"
                f"`e.response.text`: {e.response.text}\n"
                f"`url`: {url}\n"
                f"`self.cfg`: \n{str(self.cfg)}\n"
                f"`e`: \n{e}\n"
            )
            return False, ""
        except Exception as e:
            logger.error(
                "Dream API request failed, unexpected error, "
                f"`url`: {url}\n"
                f"`self.cfg`: \n{str(self.cfg)}\n"
                f"`e`: \n{e}\n"
            )
            return False, ""

    def generate_code(self, messages: PList) -> Tuple[bool, str, str]:
        try:
            ok, raw = self.plist_completion(messages)

            if not ok:
                logger.error("Failed to get response from Dream API")
                return False, "", ""

            ok, code = self.extract_code(raw)

            if not ok:
                logger.error("Failed to extract list from response")
                return False, "", raw

            return True, code, raw
        except Exception as e:
            logger.error(
                "Unexpected error while generating code with Dream,\n",
                f"`e`: \n{e}\n",
            )
            return False, "", ""

    def generate_list(self, messages: PList) -> Tuple[bool, List[str], str]:
        try:
            ok, raw = self.plist_completion(messages)

            if not ok:
                logger.error("Failed to get response from Dream API")
                return False, [], ""

            ok, items = self.extract_list(raw)

            if not ok:
                logger.error("Failed to extract list from response")
                return False, [], raw

            return True, items, raw
        except Exception as e:
            logger.error(
                "Unexpected error while generating list with Dream,\n",
                f"`e`: \n{e}\n",
            )
            return False, [], ""

    @staticmethod
    def extract_code(response: str) -> Tuple[bool, str]:
        try:
            pattern = r"```python\n([\s\S]*?)```"
            match = re.search(pattern, response, re.DOTALL)
            assert match is not None, "Code block not found"

            code_str = match.group(1)
            assert isinstance(code_str, str), "Code block is not a string"

            return True, code_str
        except Exception as e:
            logger.error("Code extraction failed: {}", e)
            return False, ""

    @staticmethod
    def extract_list(response: str) -> Tuple[bool, List[str]]:
        try:
            start, end = response.index("["), response.rindex("]") + 1
            list_str = response[start:end]
            parsed = cast(List[str], ast.literal_eval(list_str))

            assert all(isinstance(item, str) for item in parsed), (
                "All items must be strings"
            )

            return True, parsed
        except Exception as e:
            logger.error("List extraction failed: {}", e)
            return False, []
