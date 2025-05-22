from typing import Dict, Optional, TypedDict


class Message(TypedDict):
    role: str
    content: str
    meta: Optional[Dict]
