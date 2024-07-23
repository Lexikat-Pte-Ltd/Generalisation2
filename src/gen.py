import json
from typing import Any, Callable, Dict, List, Literal, cast

from jinja2 import Template
from loguru import logger
from openai import OpenAI
import requests

from src.config import DeepseekConfig, OAIConfig
from src.helper import format_ch, to_normal_plist
from src.types import Message, TaggedMessage

DEEPSEEK_TEMPLATE = """{%- if not add_generation_prompt is defined -%}
{%- set add_generation_prompt = false -%}
{%- endif -%}
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{{bos_token}}{%- if not ns.found -%}
{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
{{ message['content'] }}
    {%- else %}
        {%- if message['role'] == 'user' %}
{{'### Instruction:\\n' + message['content'] + '\\n'}}
        {%- else %}
{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}
{{'### Response:\\n'}}
{%- endif -%}"""


def _create_deepseek_genner(
    config: DeepseekConfig,
) -> Callable[[List[Message], str], str]:
    def genner(messages: List[Message], prefill="") -> str:
        template = Template(DEEPSEEK_TEMPLATE)

        prompt = template.render(
            messages=messages,
            add_generation_prompt=config.add_generation_prompt,
            bos_token=config.bos_token,
            prefill_response=prefill,
        )

        logger.debug(f"Raw prompt - \n {prompt}")

        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": config.stream,
            "format": "json",
            "raw": True,
        }

        try:
            response = requests.post(config.endpoint, json=payload)

            response.raise_for_status()

            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    return genner


def _create_oai_genner(
    client: OpenAI, config: OAIConfig
) -> Callable[[List[Message], str], str]:
    def genner(messages: List[Message], prefill="") -> str:
        try:
            response = client.chat.completions.create(
                model=config.model,
                response_format=cast(Any, config.response_format),
                messages=cast(Any, messages),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            assert isinstance(response.choices[0].message.content, str)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            raise

    return genner


def create_genner(
    backend: Literal["deepseek", "oai"],
    deepseek_config: DeepseekConfig = DeepseekConfig(),
    oai_config: OAIConfig = OAIConfig(),
    oai_client: OpenAI | None = None,
) -> Callable[[List[Message], str], str]:
    if backend == "deepseek":
        return _create_deepseek_genner(deepseek_config)
    elif backend == "oai":
        if not oai_client:
            raise ValueError("OpenAI client is required for OAI backend")
        return _create_oai_genner(oai_client, oai_config)


def gen_response(
    genner: Callable[[List[Message], str], str],
    messages: List[Message],
    expected_key: str,
    prefill: str,
) -> Any:
    try:
        logger.debug(f"Messages - \n {format_ch(messages)}")
        response = genner(messages, prefill)

        logger.debug(f"Response - \n {response}")
        response_json: Dict[str, str] = json.loads(response)

        if expected_key not in response_json:
            raise KeyError(f"Expected key '{expected_key}' not found in response")

        return response_json[expected_key]
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response")
        raise
    except KeyError as e:
        logger.error(str(e))
        raise


def gen_code(
    genner: Callable[[List[Message], str], str], messages: List[Message]
) -> str:
    response = gen_response(genner, messages, "code", "{'code':")

    assert isinstance(response, str)

    return response


def gen_strats(
    genner: Callable[[List[Message], str], str], messages: List[TaggedMessage]
) -> List[str]:
    response = gen_response(genner, to_normal_plist(messages), "list", "{'list':")

    try:
        assert isinstance(response, list)
        assert all(isinstance(item, str) for item in response)

        return response
    except AssertionError:
        logger.error("Failed making sure that generated strats are typed `List[str]`")
        raise


def unused_gen_env_code_oai(
    client: OpenAI,
    messages: List[Message],
    model="gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.5,
) -> str:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=cast(Any, messages),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    assert isinstance(response.choices[0].message.content, str)

    response_json: Dict[str, str] = json.loads(response.choices[0].message.content)

    if "code" not in response_json:
        raise TypeError("Expected key `code` in `response_json`")

    if not isinstance(response_json["code"], str):
        raise TypeError("Expected `response_json['code']` to be a `str`")

    return response_json["code"]


def unused_gen_strats_oai(
    client: OpenAI,
    messages: List[Message],
    model="gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.5,
) -> List[str]:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=cast(Any, messages),
        max_tokens=max_tokens,
        temperature=temperature,
    )

    assert isinstance(response.choices[0].message.content, str)

    response_json: Dict[str, List[str]] = json.loads(
        response.choices[0].message.content
    )

    if "list" not in response_json:
        raise TypeError("Expected key `list` in `response_json`")

    if not isinstance(response_json["list"][0], str):
        raise TypeError("Expected `response_json['list'][0]` to be a `str`")

    return response_json["list"]


def unused_gen_code_oai(
    client: OpenAI,
    messages: List[Message],
    model="gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.5,
) -> str:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=cast(Any, messages),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    assert isinstance(response.choices[0].message.content, str)

    response_json: Dict[str, List[str]] = json.loads(
        response.choices[0].message.content
    )

    if "code" not in response_json:
        raise TypeError("Expected key `code` in `response_json`")

    if not isinstance(response_json["code"], str):
        raise TypeError("Expected `response_json['code']` to be a `str`")

    return response_json["code"]


def format_list_data(data: List[Any], indent=" " * 4) -> str:
    data_ = f",\n{indent}".join(data)

    return f"[\n{indent}" + data_ + "\n]"
