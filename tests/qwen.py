from src.genner import QwenConfig, get_genner
from src.types import Message, PList

genner = get_genner("qwen", qwen_config=QwenConfig(endpoint="http://34.87.4.35:11434"))

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
