import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from loguru import logger
from unidecode import unidecode
import yaml

from src.data import EnvironmentInfo
from src.types import Message, TaggedMessage, TaggedPList


def scan_json_files_for_strat(folder_path):
    strat_list = []
    folder = Path(folder_path)

    # Iterate through all JSON files in the specified folder
    for file_path in folder.glob("*.json"):
        try:
            with file_path.open("r") as file:
                data = json.load(file)

                # Check if 'strat' key exists directly in the JSON data
                if "strat" in data:
                    strat_list.append(data["strat"])
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except IOError:
            print(f"Error reading file: {file_path.name}")

    return strat_list


def to_normal_plist(tagplist: Sequence[TaggedMessage]) -> List[Message]:
    result = [tupl[0] for tupl in tagplist]

    return result


def format_tch(tagged_chat_history: List[TaggedMessage]) -> str:
    """Use this to print tagged chat history, utilizing maximum terminal width dynamically"""
    terminal_width = shutil.get_terminal_size().columns
    to_print = "[\n"

    for chat, tag in tagged_chat_history:
        role = chat["role"]
        message = (
            chat["content"]
            .replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("'", "\\'")
        )

        # Calculate the length of the fixed part of the output
        fixed_part = f" ({{'role': '{role}', 'message': ' ... '}}, {tag}),"
        fixed_length = len(fixed_part)

        # Calculate available space for the message
        available_space = (
            terminal_width - fixed_length - 5
        )  # -5 for ellipsis and some buffer

        if len(message) <= available_space:
            formatted_message = message
        else:
            left_part = message[: available_space // 2]  # -2 for half of the ellipsis
            right_part = message[-(available_space // 2) :]  # -1 for asymmetry if odd
            formatted_message = f"{left_part} ... {right_part}"

        to_print += (
            f" ({{'role': '{role}', 'message': '{formatted_message}'}}, {tag}),\n"
        )

    to_print += "]"
    return to_print


def format_ch(chat_history: List[Message]) -> str:
    """Use this to print chat history, utilizing maximum terminal width dynamically"""
    terminal_width = shutil.get_terminal_size().columns
    to_print = "[\n"

    for chat in chat_history:
        role = chat["role"]
        message = chat["content"].replace("\n", " ").replace("'", "\\'")

        # Calculate the length of the fixed part of the output
        fixed_part = f"  {{'role': '{role}', 'message': ''}},"
        fixed_length = len(fixed_part)

        # Calculate available space for the message
        available_space = (
            terminal_width - fixed_length - 5
        )  # -5 for ellipsis and some buffer

        if len(message) <= available_space:
            formatted_message = message
        else:
            left_part = message[
                : available_space // 2 - 2
            ]  # -2 for half of the ellipsis
            right_part = message[
                -(available_space // 2 - 1) :
            ]  # -1 for asymmetry if odd
            formatted_message = f"{left_part}...{right_part}"

        to_print += f"  {{'role': '{role}', 'message': '{formatted_message}'}},\n"

    to_print += "]"
    return to_print


def format_eih(environment_info_history: List[EnvironmentInfo]) -> str:
    terminal_width = shutil.get_terminal_size().columns
    to_print = "[\n"

    for env_info in environment_info_history:
        info_str = str(env_info)
        lines = info_str.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Calculate available space for each line
            if line.startswith("From"):
                indent = "    "
            else:
                indent = "        "
            available_space = terminal_width - len(indent)

            if len(line) <= available_space:
                formatted_lines.append(f"{indent}{line}")
            else:
                # Split long lines
                words = line.split()
                current_line = indent
                for word in words:
                    if len(current_line) + len(word) + 1 <= terminal_width:
                        current_line += word + " "
                    else:
                        formatted_lines.append(current_line.rstrip())
                        current_line = (
                            indent + "  " + word + " "
                        )  # Extra indentation for continuation
                if current_line.strip():
                    formatted_lines.append(current_line.rstrip())

        to_print += "  EnvironmentInfo(\n"
        to_print += "\n".join(formatted_lines)
        to_print += "\n  ),\n"

    to_print += "]"
    return to_print


def sanitize_code(code: str):
    code = unidecode(code)

    code = code.replace(""", "'").replace(""", "'").replace('"', '"').replace('"', '"')

    # Replace full-width vertical bar with standard vertical bar
    code = code.replace("｜", "|")

    # Replace other potential problematic characters
    code = code.replace("…", "...")

    # Use regex to find and replace any remaining non-ASCII characters
    code = re.sub(r"[^\x00-\x7F]+", "", code)

    return code


def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


def process_old_tch(old_tch: TaggedPList) -> List[Dict]:
    new_tch = []
    for item in old_tch:
        if isinstance(item, list) and len(item) == 2:
            new_tch.append({"message": item[0], "tag": item[1]})
        elif isinstance(item, dict) and "message" in item and "tag" in item:
            new_tch.append(item)
        else:
            print(f"Unexpected format: {item}")

    return new_tch


def represent_multiline_str(dumper: yaml.Dumper, data: str) -> yaml.nodes.ScalarNode:
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)
