from typing import Dict, NamedTuple


class DeepseekConfig(NamedTuple):
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "deepseek-coder:6.7b"
    stream: bool = False
    add_generation_prompt: bool = False
    bos_token: str = ""
    prefill_response: str = ""


class OAIConfig(NamedTuple):
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    response_format: Dict[str, str] = {"type": "json_object"}
    temperature: float = 0.5
