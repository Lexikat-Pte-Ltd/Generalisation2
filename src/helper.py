from loguru import logger
from typing import List, Sequence
from src.types import TaggedMessage, Message


def to_normal_plist(tagplist: Sequence[TaggedMessage]) -> List[Message]:
    result = [tupl[0] for tupl in tagplist]

    return result


def format_tch(tagged_chat_history: List[TaggedMessage]) -> str:
    """Use this to print tagged chat history"""

    to_print = "[\n"

    for chat, tag in tagged_chat_history:
        role = chat["role"]
        message = chat["content"].replace("\n", "")

        to_print += "\t"
        to_print += f"({{'role': '{role}', 'message': '{message[:10]} ... {message[-10:]}'}}, {tag})"
        to_print += ",\n"

    to_print += "]"

    return to_print


def format_ch(chat_history: List[Message]) -> str:
    """Use this to print chat history"""

    to_print = "[\n"

    for chat in chat_history:
        role = chat["role"]
        message = chat["content"].replace("\n", "")

        to_print += "\t"
        to_print += (
            f"{{'role': '{role}', 'message': '{message[:10]} ... {message[-10:]}'}}"
        )
        to_print += ",\n"

    to_print += "]"

    return to_print
