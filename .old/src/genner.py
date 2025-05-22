# import ast
# import json
# from abc import ABC, abstractmethod
# import re
# from typing import Any, Dict, List, Literal, Tuple, cast

# from httpx import request
# import httpx
# from jinja2 import Template
# from loguru import logger
# from openai import OpenAI
# from anthropic import Anthropic


# from src.config import (
#   OllamaConfig,
#   DeepseekConfig,
#   OAIConfig,
#   QwenConfig,
#   WizardCoderConfig,
#   ClaudeConfig,
# )
# from src.helper import to_normal_plist
# from src.types import Message, TaggedMessage

# def format_list_data(data: List[Any], indent=" " * 4) -> str:
#   data_ = f",\n{indent}".join(data)

#   return f"[\n{indent}" + data_ + "\n]"
