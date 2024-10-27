from typing import Dict, NamedTuple


class DeepseekConfig(NamedTuple):
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "deepseek-coder-j:latest"
    stream: bool = False
    add_generation_prompt: bool = True
    bos_token: str = "<|begin_of_sentence|>"
    bos_token_2: str = "<｜begin▁of▁sentence｜>"
    eos_token: str = "<|end_of_sentence|>"
    eos_token_2: str = "<｜end▁of▁sentence｜>"
    eot_token: str = "<|EOT|>"
    eot_token_2: str = "<｜EOT｜>"
    vertical_bar: str = "｜"


class WizardCoderConfig(NamedTuple):
    endpoint: str = "http://localhost:11434/api/generate"
    model: str = "wizardcoder:latest"
    stream: bool = False
    add_generation_prompt: bool = True


class OAIConfig(NamedTuple):
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    response_format: Dict[str, str] = {"type": "json_object"}
    temperature: float = 0.5

class ClaudeConfig(NamedTuple):
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 5
    stream: bool = False
