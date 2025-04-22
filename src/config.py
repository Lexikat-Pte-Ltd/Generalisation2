from abc import ABC
from typing import Dict, NamedTuple


class OllamaConfig(ABC):
    name: str
    endpoint: str = "http://localhost:11434/api/chat"
    model: str
    stream: bool


class DeepseekConfig(OllamaConfig):
    name: str = "Ollama DeepSeek"
    model: str = "deepseek-coder:6.7b"
    stream: bool = False


class WizardCoderConfig(OllamaConfig):
    name: str = "Ollama WizardCoder"
    model: str = "wizardcoder:7b-python"
    stream: bool = False


class QwenConfig(OllamaConfig):
    name: str = "Ollama Qwen"
    model: str = "qwen2.5-coder:latest"
    model_uncensored: str = "qwen-uncensored:latest"

    stream: bool = False


class OAIConfig(NamedTuple):
    name: str = "OpenAI"
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.5


class ClaudeConfig(NamedTuple):
    name: str = "Anthropic Claude"
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 5
    stream: bool = False


class DreamConfig(NamedTuple):
    """Configuration for the Dream API generator."""

    # API connection settings
    base_url: str = "http://34.124.150.248:6969"
    timeout: int = 120

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95

    # Algorithm settings
    steps: int = 10
    alg: str = "entropy"
    alg_temp: float = 0
