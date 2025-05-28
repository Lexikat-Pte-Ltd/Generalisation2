from abc import ABC
from typing import Dict, NamedTuple
from pydantic import BaseModel


class OllamaConfig(ABC, BaseModel):
    name: str
    endpoint: str = "localhost:11434"
    model: str
    stream: bool

class QwenConfig(OllamaConfig):
    name: str = "Ollama Qwen"
    model: str = "qwen2.5-coder:latest"
    _model_uncensored: str = "qwen-uncensored:latest"
    stream: bool = False

class DreamConfig(NamedTuple):
    """Configuration for the Dream API generator."""

    # API connection settings
    base_url: str = "http://34.87.4.35:6969"
    # Allow overriding via DREAM_BASE_URL; default to localhost if unset
    # base_url: str = os.getenv("DREAM_BASE_URL", "http://localhost:6969")
    timeout: int = 120

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.8
    steps: int = 64

    top_k: int = 20

    alg: str = "origin"
    alg_temp: float = 0.3
