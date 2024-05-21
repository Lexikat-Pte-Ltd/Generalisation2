import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
from pprint import pformat

from docker import DockerClient
from openai import OpenAI
from loguru import logger

from src.container import run_code_in_con, safe_detect_env, safe_folderlist_bfs
from src.data import EnvironmentInfo

SYSTEM_TEMPLATE = """
You are an agent working in a operating system with following information encoded in \"Information\" XML tag. Those information are
<Informations>
- You are working in `{in_con_path}` path.
- The basic system information can be described in JSON, encoded in \"BasicEnvInfo\" XML tag, these are
<BasicEnvInfo>
{env_info}
</BasicEnvInfo>
</Information>

You are tasked with 3 distinct tasks, encoded in \"Tasks\" XML tag, these are
<Tasks>
- Writing a code that returns necessary information about the operating system to perform the other tasks you are tasked to perform.
- Writing set of strategies to free up spaces inside the operating system you are working with.
- Writing codes to perform strategies you have originally came up with to free storage spaces inside of the operating system.
</Tasks>
""".strip()


TASKGEN_TASK_TEMPLATE = """
Given the previous context, generate a few speific potential strategies as an AI agent to achieve freeing up some disk space for the directory `{in_con_path}`. 
Please generate the task in JSON list exactly formatted like {{"list": ["strategy1", "strategy2", ... "strategyN"]}}. 
Wrapped in the \"Strategies\" XML tag, the strategies are <Strategies>\n
""".strip()


def prepare_taskgen_prompt_openai(
    in_con_path: str | Path,
    task_template=TASKGEN_TASK_TEMPLATE,
) -> Tuple[List[Dict[str, str]], str]:
    task_prompt = task_template.format(
        in_con_path=in_con_path  #
    )

    return [
        {"role": "user", "content": task_prompt},
    ], "taskgen"


ENV_CODEGEN_TEMPLATE = """
Given the previous context, generate python code for obtaining environment description for the strategy \"{task_description}\" working in the directory of {in_con_path}.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
The code are expected to print out string to stdout containing environment information information.
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`for.
Wrapped in the \"EnvCodeGen\" XML tag, the code is <EnvCodeGen>\n
""".strip()


def prepare_env_codegen_prompt_openai(
    in_con_path: str | Path,
    task_description="Free spaces on all of the directory of the operating system",
    env_codegen_template=ENV_CODEGEN_TEMPLATE,
) -> Tuple[List[Dict[str, str]], str]:
    env_codegen_prompt = env_codegen_template.format(
        task_description=task_description, in_con_path=in_con_path
    )

    return [
        {"role": "user", "content": env_codegen_prompt},
    ], "env_codegen"


CODEGEN_TASK_TEMPLATE = """
Given the previous context, generate python code for the task \"{task_description}\".
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code :\n
""".strip()


def prepare_task_codegen_prompt_openai(
    task_description: str,
    code_template=CODEGEN_TASK_TEMPLATE,
) -> Tuple[List[Dict[str, str]], str]:
    code_prompt = code_template.format(
        task_description=task_description  #
    )

    return [
        {"role": "user", "content": code_prompt},
    ], "task_codegen"


REGEN_TEMPLATE = """
Output of the code above is after running it on {run_context} is:
{error_context}
Please improve upon the code you have written by keeping in mind the error output above where task is {task_description}.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code :\n
""".strip()


def prepare_regenerate_prompts_openai(
    task_description: str,
    prev_code: str,
    error_context: str,
    run_context: str,
    regen_template=REGEN_TEMPLATE,
) -> List[Dict[str, str]]:
    regen_prompts = regen_template.format(
        task_description=task_description,
        error_context=error_context,
        run_context=run_context,
    )

    return [
        {"role": "assistant", "content": prev_code},
        {"role": "user", "content": regen_prompts},
    ]


IMPROVE_TEMPLATE = """
Results of the code above is after running it on {run_context} are:
{newest_env_info}
With the difference of (Previous - Current):
{env_info_diff}
Please improve upon the code you have written by keeping in mind the error output above where task is {task_description}.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code:\n
""".strip()


def prepare_improve_prompt_openai(
    prev_code: str,
    newest_env_info: EnvironmentInfo,
    diff_env: EnvironmentInfo,
    task_description: str,
    run_context: str,
    improve_template=IMPROVE_TEMPLATE,
) -> List[Dict[str, str]]:
    improve_prompt = improve_template.format(
        run_context=run_context,
        newest_env_info=newest_env_info.model_dump_json(indent=4),
        env_info_diff=diff_env.model_dump_json(indent=4),
        task_description=task_description,
    )

    return [
        {"role": "assistant", "content": prev_code},
        {"role": "user", "content": improve_prompt},
    ]


def gen_skill_sets_openai(
    client: OpenAI,
    messages: List[Dict[str, str]],
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


def gen_skill_code_openai(
    client: OpenAI,
    messages: List[Dict[str, str]],
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
