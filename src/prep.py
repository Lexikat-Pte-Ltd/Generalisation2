from pathlib import Path
from typing import List

from src.data import EnvironmentInfo
from src.types import TaggedMessage

# {"role": "system"}
SYSTEM_TEMPLATE = """
You are an agent working in a operating system with following information encoded in \"Informations\" XML tag. Those information are
<Informations>
- You are working in `{in_con_path}` path.
</Information>

You are tasked with 3 distinct tasks, encoded in \"Tasks\" XML tag, these are
<Tasks>
- Writing a code that returns necessary information about the operating system to perform the other tasks you are tasked to perform.
- Writing set of strategies to free up spaces inside the operating system you are working with.
- Writing codes to perform these strategies you have originally came up with to free storage spaces inside of the operating system.
</Tasks>
""".strip()


# Used by basic and env agent
def prep_system_plist_oai(
    in_con_path: str,
    system_template=SYSTEM_TEMPLATE,
    tag="system_prompt",
) -> List[TaggedMessage]:
    system_prompt = system_template.format(
        in_con_path=in_con_path,  #
    )

    return [
        ({"role": "system", "content": system_prompt}, tag),
    ]


# {"role": "user"}
BASIC_ENV_INFOS_INCLUSION_TEMPLATE = """
Here's also an additional environment information that you can use to aid you in this process, encoded in \"EnvInfo\" XML tag, those are
<BasicEnvInfo>
{basic_env_info}
</BasicEnvInfo>
"""


# Used by basic and env agent
def prep_basic_env_plist_oai(
    basic_env_info: str,
    inclusion_template=BASIC_ENV_INFOS_INCLUSION_TEMPLATE,
    tag="basic_env_facilitation",
):
    """Sets of chat history to append to env's and basic's agent chat history

    Args:
        basic_env_info (str): Initial or basic environment info
        inclusion_template (str, optional): Prompt template. Defaults to BASIC_ENV_INFOS_INCLUSION_TEMPLATE.
        tag (str, optional): Tag to identify the single chat history. Defaults to "basic_env_facilitation".

    Returns:
        _type_: _description_
    """
    inclusion_prompt = inclusion_template.format(basic_env_info=basic_env_info)

    return [
        ({"role": "user", "content": inclusion_prompt}, tag),
    ]


SPECIAL_ENV_INFOS_INCLUSION_TEMPLATE = """
Here's also additional environment informations that you can use to aid you in this process, encoded in \"SpecialEnvInfos\" XML tag, those are
<SpecialEnvInfos>
{special_env_infos}
</SpecialEnvInfos>
"""

SPECIAL_ENV_INFO_TEMPLATE = """
<SpecialEnvInfo>
{special_env_info}
</SpecialEnvInfo>
"""


# Used by basic agent
def prep_special_env_plist_oai(
    special_env_infos: List[str],
    inclusion_template=SPECIAL_ENV_INFOS_INCLUSION_TEMPLATE,
    singular_env_info_template=SPECIAL_ENV_INFO_TEMPLATE,
    tag="special_env_facilitation",
) -> List[TaggedMessage]:
    wrapped_env_infos = []

    for env_info in special_env_infos:
        wrapped_env_infos.append(
            singular_env_info_template.format(special_env_info=env_info)
        )

    inclusion_prompt = inclusion_template.format(
        special_env_infos="".join(wrapped_env_infos)
    )

    return [
        ({"role": "user", "content": inclusion_prompt}, tag),
    ]


STRATGEN_TASK_TEMPLATE = """
Given the previous context, generate a few specific potential strategies as an AI agent to achieve freeing up some disk space for the directory `{in_con_path}`. 
Please generate the task in JSON list exactly formatted like {{"list": ["strategy1", "strategy2", ... "strategyN"]}}. 
""".strip()


# Used by basic agent
def prep_stratgen_plist_oai(
    in_con_path: str | Path, strat_template=STRATGEN_TASK_TEMPLATE, tag="strat_request"
) -> List[TaggedMessage]:
    strat_prompt = strat_template.format(
        in_con_path=in_con_path  #
    )

    return [
        ({"role": "user", "content": strat_prompt}, tag),
    ]


SPECIAL_ENV_CODEGEN_TEMPLATE = """
Given the previous context, generate python code for obtaining special environment description that is going to assists you with tasks defined in the system prompt.
You are working in the directory of {in_con_path}.
Be original, unique, and create other information that has yet existed in the previous environment info.
Also use default python library and nothing else.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
The code are expected to print out string to stdout containing environment information.
If there is another code you have previously generated, be different and unique to the previous one.
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`for without installing another new library.
""".strip()


# Used by env agent
def prep_special_env_codegen_plist_oai(
    in_con_path: str | Path,
    special_env_codegen_template=SPECIAL_ENV_CODEGEN_TEMPLATE,
) -> List[TaggedMessage]:
    env_codegen_prompt = special_env_codegen_template.format(in_con_path=in_con_path)

    return [
        ({"role": "user", "content": env_codegen_prompt}, "env_codegen_request"),
    ]


CODEGEN_TASK_TEMPLATE = """
Given the previous context, generate python code for the task \"{task_description}\".
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code :\n
""".strip()


# Used by basic agent
def prep_strat_codegen_plist_oai(
    task_description: str,
    code_template=CODEGEN_TASK_TEMPLATE,
) -> List[TaggedMessage]:
    code_prompt = code_template.format(
        task_description=task_description  #
    )

    return [
        ({"role": "user", "content": code_prompt}, "regen_request"),
    ]


REGEN_TEMPLATE = """
Output of the code above is after running it on {run_context} is:
{error_context}
Please improve upon the code you have written by keeping in mind the error output above where task is {task_description}.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code :\n
""".strip()


# Used by basic and env agent
def prep_regen_plist_oai(
    task_description: str,
    prev_code: str,
    error_context: str,
    run_context: str,
    regen_template=REGEN_TEMPLATE,
) -> List[TaggedMessage]:
    regen_prompts = regen_template.format(
        task_description=task_description,
        error_context=error_context,
        run_context=run_context,
    )

    return [
        ({"role": "assistant", "content": prev_code}, "failure_code"),
        ({"role": "user", "content": regen_prompts}, "regen_request"),
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


# Used by basic and env agent
def prep_improve_plist_oai(
    prev_code: str,
    newest_env_info: EnvironmentInfo,
    diff_env: EnvironmentInfo,
    task_description: str,
    run_context: str,
    improve_template=IMPROVE_TEMPLATE,
) -> List[TaggedMessage]:
    improve_prompt = improve_template.format(
        run_context=run_context,
        newest_env_info=newest_env_info.model_dump_json(indent=4),
        env_info_diff=diff_env.model_dump_json(indent=4),
        task_description=task_description,
    )

    return [
        ({"role": "assistant", "content": prev_code}, "successful_code"),
        ({"role": "user", "content": improve_prompt}, "improve_request"),
    ]
