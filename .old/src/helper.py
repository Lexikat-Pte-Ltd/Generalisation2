import difflib
import json
import random
import re
import select
import shutil
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence

import yaml
from unidecode import unidecode

from src.data import EnvironmentInfo
from src.types import PList, TaggedPList


def scan_json_files_for_strat(folder_path) -> List[str]:
    strat_list = []
    folder = Path(folder_path)

    # Iterate through all JSON files in the specified folder
    for file_path in folder.glob("*.json"):
        try:
            with file_path.open("r") as file:
                data = json.load(file)

                # Check if 'strat' key exists directly in the JSON data
                if "main_strat_agent" in data:
                    if "strats" in data["main_strat_agent"]:
                        strat_list.extend(data["main_strat_agent"]["strats"])

        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path.name}")
        except IOError:
            print(f"Error reading file: {file_path.name}")

    if len(strat_list) > 10:
        strat_list = random.sample(strat_list, 10)

    return strat_list


def format_tch(tch: TaggedPList) -> str:
    """Use this to print tagged chat history, utilizing maximum terminal width dynamically"""
    terminal_width = shutil.get_terminal_size().columns
    to_print = "[\n"

    for tagged_message in tch.messages:
        role = tagged_message.message.role
        message = (
            tagged_message.message.content.replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("'", "\\'")
        )

        # Calculate the length of the fixed part of the output
        fixed_part = (
            f" ({{'role': '{role}', 'message': ' ... '}}, {tagged_message.tag}),"
        )
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

        to_print += f" ({{'role': '{role}', 'message': '{formatted_message}'}}, {tagged_message.tag}),\n"

    to_print += "]"
    return to_print


def format_ch(chat_history: PList) -> str:
    """Use this to print chat history, utilizing maximum terminal width dynamically"""
    terminal_width = shutil.get_terminal_size().columns
    to_print = "[\n"

    for message in chat_history.messages:
        role = message.role
        message = message.content.replace("\n", " ").replace("'", "\\'")

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


def represent_multiline_str(dumper: yaml.Dumper, data: str) -> yaml.nodes.ScalarNode:
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def normalize_line_endings(code: str) -> str:
    """Normalize different types of line endings to \n"""
    return code.replace("\r\n", "\n").replace("\r", "\n")


def normalize_escapes(s: str) -> str:
    return s.encode("utf-8").decode("unicode_escape")


def get_alpha_first_sentences(s: str) -> str:
    words = s.split(" ")
    return "".join([c.lower() for c in words[0] if c.isalpha()])


def get_readable_stratname_format(s: str) -> str:
    words = s.split(" ")
    first_word = words[0].lower()
    alpha_words = [word.lower() for word in words[1:] if word.isalpha()]

    return f"{first_word}_{''.join(alpha_words)[:20]}"


def get_code_diff(
    old_code: str, new_code: str, ignore_line_endings: bool = True
) -> List[str]:
    old_code = normalize_escapes(old_code)
    new_code = normalize_escapes(new_code)

    if ignore_line_endings:
        old_code = normalize_line_endings(old_code)
        new_code = normalize_line_endings(new_code)

    differ = difflib.SequenceMatcher(None, old_code, new_code)
    changes = []

    for opcode, old_start, old_end, new_start, new_end in differ.get_opcodes():
        if opcode != "equal":
            old_substr = old_code[old_start:old_end]
            new_substr = new_code[new_start:new_end]

            # Skip if the only thing that are different are line endings
            if ignore_line_endings and normalize_line_endings(
                old_substr
            ) == normalize_line_endings(new_substr):
                continue

            if opcode == "replace":
                # Make line endings visible in output
                old_display = repr(old_substr)[1:-1]
                new_display = repr(new_substr)[1:-1]

                changes.append(f"Changed '{old_display}' to '{new_display}'")
            elif opcode == "delete":
                changes.append(f"Deleted '{old_substr}'")
            elif opcode == "insert":
                changes.append(f"Inserted '{new_substr}'")

    return changes


def timed_input(prompt, timeout=3):
    print(prompt, end="", flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline().strip()
    else:
        print("\nContinuing...")
        return None


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def int_to_ordinal(number: int):
    """
    Convert an integer to its ordinal representation.

    Args:
        number (int): The integer to convert

    Returns:
        str: The ordinal representation (e.g., '1st', '2nd', '3rd', '4th')
    """
    if number <= 0:
        raise ValueError("Expected a positive integer")

    # Special cases for 11, 12, 13
    if 11 <= (number % 100) <= 13:
        suffix = "th"
    else:
        # Handle the general cases
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")

    return f"{number}{suffix}"


# Examples
if __name__ == "__main__":
    test_numbers = [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 102, 111, 112, 1001]

    for num in test_numbers:
        print(f"{num} -> {int_to_ordinal(num)}")
