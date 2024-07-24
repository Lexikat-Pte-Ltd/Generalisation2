import shutil
from loguru import logger
from typing import List, Sequence
from src.data import EnvironmentInfo
from src.types import TaggedMessage, Message


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
