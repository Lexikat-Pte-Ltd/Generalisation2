from src.config import (
  DeepseekConfig,
  OAIConfig,
  WizardCoderConfig,
  QwenConfig,
  ClaudeConfig,
)
from .Base import Genner
from .Deepseek import DeepseekGenner
from .Qwen import QwenGenner
from .WizardCoder import WizardCoderGenner
from .OAI import OAIGenner
from .Claude import ClaudeGenner
from openai import OpenAI
from anthropic import Anthropic

__all__ = ["get_genner"]


class BackendException(Exception):
  pass


class OaiBackendException(Exception):
  pass


class ClaudeBackendException(Exception):
  pass


def get_genner(
  backend: str,
  deepseek_config: DeepseekConfig = DeepseekConfig(),
  oai_config: OAIConfig = OAIConfig(),
  wizard_config: WizardCoderConfig = WizardCoderConfig(),
  qwen_config: QwenConfig = QwenConfig(),
  claude_config: ClaudeConfig = ClaudeConfig(),
  oai_client: OpenAI | None = None,
  claude_client: Anthropic | None = None,
) -> Genner:
  """
  Get a genner instance based on the backend.

  Args:
      backend (str): The backend to use.
      deepseek_config (DeepseekConfig, optional): The configuration for the Deepseek backend. Defaults to DeepseekConfig().
      oai_config (OAIConfig, optional): The configuration for the OpenAI backend. Defaults to OAIConfig().
      wizard_config (WizardCoderConfig, optional): The configuration for the WizardCoder backend. Defaults to WizardCoderConfig().
      qwen_config (QwenConfig, optional): The configuration for the Qwen backend. Defaults to QwenConfig().
      claude_config (ClaudeConfig, optional): The configuration for the Claude backend. Defaults to ClaudeConfig().
      oai_client (OpenAI | None, optional): The OpenAI client. Defaults to None.
      claude_client (Anthropic | None, optional): The Anthropic client. Defaults to None.

  Raises:
      BackendException: If the backend is not supported.
      OaiBackendException: If the OpenAI client is required for the OAI backend but not provided.
      ClaudeBackendException: If the Anthropic client is required for the Claude backend but not provided.

  Returns:
      Genner: The genner instance.
  """
  available_backends = ["deepseek", "qwen", "wizardcoder", "oai", "claude"]

  if backend == "deepseek":
    return DeepseekGenner(deepseek_config)
  elif backend == "qwen":
    return QwenGenner(qwen_config)
  elif backend == "wizardcoder":
    return WizardCoderGenner(wizard_config)
  elif backend == "oai":
    if not oai_client:
      raise OaiBackendException(
        "Using backend 'oai', OpenAI client is required for OAI backend"
      )
    return OAIGenner(oai_client, oai_config)
  elif backend == "claude":
    if not claude_client:
      raise ClaudeBackendException("Anthropic client is required for Claude backend")
    return ClaudeGenner(claude_client, claude_config)

  raise BackendException(
    f"Unsupported backend: {backend}, available backends: {', '.join(available_backends)}"
  )
