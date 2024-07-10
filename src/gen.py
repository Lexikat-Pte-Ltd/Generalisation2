from openai import OpenAI
import json
from typing import Any, Dict, List, cast

from src.types import Message


def gen_env_code_oai(
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


def gen_strategies_oai(
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


def gen_code_oai(
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
