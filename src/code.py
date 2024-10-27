import ast
from typing import Tuple
from src.data import EnvironmentInfo

from loguru import logger


def is_valid_code_ast(code: str) -> Tuple[bool, str]:
  try:
    ast.parse(code)
  except SyntaxError as e:
    return False, f"Syntax Error: {e}"
  return True, ""


def is_valid_code_compiler(code: str) -> Tuple[bool, str]:
  try:
    compile(code, "<string>", "exec")

    return True, ""
  except SyntaxError as e:
    return False, f"Syntax Error: {e}"


def is_working(
  prev_env_info: EnvironmentInfo, cur_env_info: EnvironmentInfo, storage_only=True
):
  # If previous is more then something has been deleted
  size_diff = prev_env_info.available_storage - cur_env_info.available_storage
  if size_diff < 0:
    logger.info(f"size_diff {size_diff}")
    return True

  return False


def process_response_code(code: str):
  # cleaned_code = code[code.find("```python") + 9 : code.rfind("```")]
  # escaped_code = cleaned_code.replace("'", "'\"'\"'")

  return code
