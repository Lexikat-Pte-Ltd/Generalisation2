from typing import cast
from src.genner import get_genner
from src.genner.Dream import DreamGenner
from src.types import Message, PList

genner = cast(DreamGenner, get_genner("dream"))

success, response = genner.plist_completion(
    messages=PList(
        [
            Message(
                role="user",
                content="What is the capital of France?",
            )
        ]
    )
)

print("Success:", success)
print("Response:", response)
