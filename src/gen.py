import ast
import json
from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Literal, Tuple, cast

import requests
from jinja2 import Template
from loguru import logger
from openai import OpenAI
from unidecode import unidecode

from src.config import DeepseekConfig, OAIConfig
from src.helper import to_normal_plist
from src.types import Message, TaggedMessage

DEEPSEEK_TEMPLATE = """{%- set ns = namespace(found=false, last_role=None) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
{{ message['content'] }}
    {%- else %}
        {%- if message['role'] == 'user' %}
            {%- if ns.last_role == 'assistant' %}
{{'### Instruction:\n' + message['content'] + '\n'}}
            {%- else %}
{{ message['content'] + '\n' }}
            {%- endif %}
        {%- elif message['role'] == 'assistant' %}
{{'### Response:\n' + message['content'] + '\n'}}
        {%- endif %}
    {%- endif %}
    {%- set ns.last_role = message['role'] %}
{%- endfor %}
### Response:
{% if force_python %}
```python
{% endif %}
""".strip()


class Genner(ABC):
    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    def generate_code(self, messages: List[Message]) -> Tuple[str, str]:
        pass

    @abstractmethod
    def generate_list(self, messages: List[TaggedMessage]) -> Tuple[List[str], str]:
        pass


class DeepseekGenner(Genner):
    def __init__(self, config: DeepseekConfig):
        self.config = config

    def generate(self, messages: List[Message], force_python=False) -> str:
        template = Template(DEEPSEEK_TEMPLATE)
        prompt = template.render(
            messages=messages,
            force_python=force_python,
            add_generation_prompt=self.config.add_generation_prompt,
            bos_token=self.config.bos_token,
        )

        logger.debug(f"Raw prompt - \n {prompt}")

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": self.config.stream,
        }

        try:
            response = requests.post(self.config.endpoint, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    @staticmethod
    def _extract_code(response: str) -> str:
        # Extract code from the response
        regex_pattern = r"```python\n([\s\S]*?)```"
        code_match = re.search(regex_pattern, response, re.DOTALL)
        assert code_match is not None

        code_string = code_match.group(1)
        assert code_string is not None

        return code_string

    def generate_code(self, messages: List[Message]) -> Tuple[str, str]:
        while True:
            try:
                raw_response = self.generate(messages, force_python=False)

                processed_code = self._extract_code(raw_response)
                processed_code = (
                    processed_code
                    .replace(self.config.bos_token, "")
                    .replace(self.config.bos_token_2, "")
                    .replace(self.config.eos_token, "")
                    .replace(self.config.eos_token_2, "")
                    .replace(self.config.eot_token, "")
                    .replace(self.config.eot_token_2, "")
                    .replace(self.config.vertical_bar, "|")
                    .strip()
                )

                logger.info(f"Processed code - \n{processed_code}")

                return processed_code, raw_response
            except Exception as e:
                logger.error(f"An error while generating code occured: {e}")
                logger.error("Retrying...")

    @staticmethod
    def _extract_list(response: str) -> List[str]:
        start = response.index("[")
        end = response.rindex("]") + 1
        list_string = response[start:end]

        # Parse the string to a Python list
        processed_list = ast.literal_eval(list_string)

        assert isinstance(processed_list, list)
        assert all(isinstance(item, str) for item in processed_list)

        return processed_list

    def generate_list(self, messages: List[TaggedMessage]) -> Tuple[List[str], str]:
        # Deepseek-specific list generation logic
        while True:
            try:
                raw_response = self.generate(to_normal_plist(messages))
                processed_list = self._extract_list(raw_response)

                return processed_list, raw_response
            except Exception as e:
                logger.error(f"An error while generating list occured: {e}")
                logger.error("Retrying...")


class OAIGenner(Genner):
    def __init__(self, client: OpenAI, config: OAIConfig):
        self.client = client
        self.config = config

    def generate(self, messages: List[Message]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                response_format={"type": "json_object"},
                messages=cast(Any, messages),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            assert isinstance(response.choices[0].message.content, str)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API request failed: {str(e)}")
            raise

    def generate_code(self, messages: List[Message]) -> Tuple[str, str]:
        raw_response = self.generate(messages)
        response_json = json.loads(raw_response)
        processed_code = response_json.get("code", "")

        return processed_code, raw_response

    def generate_list(self, messages: List[TaggedMessage]) -> Tuple[List[str], str]:
        raw_response = self.generate(to_normal_plist(messages))
        response_json = json.loads(raw_response)
        list_response = response_json.get("list", [])

        if isinstance(list_response, str):
            processed_list = json.loads(list_response)
        else:
            processed_list = list_response

        assert isinstance(processed_list, list)
        assert all(isinstance(item, str) for item in processed_list)

        return processed_list, raw_response


def get_genner(
    backend: Literal["deepseek", "oai"],
    deepseek_config: DeepseekConfig = DeepseekConfig(),
    oai_config: OAIConfig = OAIConfig(),
    oai_client: OpenAI | None = None,
) -> Genner:
    if backend == "deepseek":
        return DeepseekGenner(deepseek_config)
    elif backend == "oai":
        if not oai_client:
            raise ValueError("OpenAI client is required for OAI backend")
        return OAIGenner(oai_client, oai_config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


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
