from typing import Dict, TypedDict, NotRequired, Any


class Message(TypedDict):
    role: str
    content: str
    meta: NotRequired[Dict[str, Any]]
