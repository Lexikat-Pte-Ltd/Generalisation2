from pathlib import Path
from typing import List

from src.data import EnvironmentInfo
from src.types import TaggedMessage

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

SYSTEM_PLIST_TAG = "system_plist"


def get_system_plist(
    in_con_path: str,
    tag=SYSTEM_PLIST_TAG,
) -> List[TaggedMessage]:
    """(System, Env, Basic)

    Get system prompt plist (single) that precedes all other tagged chat history.

    ```
    On EnvAgent and BasicAgent :
    [
        > ({"role": "system", "content": "..."}, "system"),
        ...,
        ...,
    ]
    ```

    Args:
        in_con_path (str): In container path.
        tag (str, optional): Tag to identify the only message in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    system_prompt = SYSTEM_TEMPLATE.format(
        in_con_path=in_con_path,  #
    )

    return [
        ({"role": "system", "content": system_prompt}, tag),
    ]


PLURAL_BASIC_ENV_INFO_INCLUSION_TEMPLATE = """
Here's also an additional environment information that you can use to aid you in this process, encoded in \"EnvInfo\" XML tag, those are
<BasicEnvInfos>
{basic_env_infos}
</BasicEnvInfos>
""".strip()

SINGULAR_BASIC_ENV_INFO_INCLUSION_TEMPLATE = """
<BasicEnvInfo>
{basic_env_info}
</BasicEnvInfo>
""".strip()

BASIC_ENV_PLIST_TAG = "get_basic_env_plist"


def get_basic_env_plist(
    bs_eih: List[EnvironmentInfo],
    tag=BASIC_ENV_PLIST_TAG,
    max_count=5,
) -> List[TaggedMessage]:
    """(User, EnvAgent, CommonAgent, ContextProvider)

    Get basic env plist for basic environment inclusion.

    ```
    On EnvAgent :
    [
        ({"role": "system", "content": "..."}, "get_system_plist"),
        > ({"role": "user", "content": "..."}, "get_basic_env_plist"),
        ({"role": "user", "content": "..."}, "get_special_env_plist"),
        ...,
    ]
    ```

    Args:
        bs_eih (str): Initial or basic environment info history.
        tag (str, optional): Tag to identify the only message in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    if len(bs_eih) > 1:
        inner_env_infos = []

        for env_info in bs_eih:
            inner_env_infos.append(
                SINGULAR_BASIC_ENV_INFO_INCLUSION_TEMPLATE.format(
                    basic_env_info=str(env_info)
                )
            )

        outer_env_infos = PLURAL_BASIC_ENV_INFO_INCLUSION_TEMPLATE.format(
            basic_env_infos="".join(inner_env_infos[-max_count:])
        )
    else:
        outer_env_infos = SINGULAR_BASIC_ENV_INFO_INCLUSION_TEMPLATE.format(
            basic_env_info=bs_eih[0]
        )

    return [
        ({"role": "user", "content": outer_env_infos}, tag),
    ]


PLURAL_SPECIAL_ENV_INFOS_INCLUSION_TEMPLATE = """
Here's also additional environment informations that you can use to aid you in this process, encoded in \"SpecialEnvInfos\" XML tag, those are
<SpecialEnvInfos>
{special_env_infos}
</SpecialEnvInfos>
""".strip()

SINGULAR_SPECIAL_ENV_INFO_INCLUSION_TEMPLATE = """
<SpecialEnvInfo>
{special_env_info}
</SpecialEnvInfo>
""".strip()

SPECIAL_ENV_PLIST_TAG = "get_special_env_plist"


# Used by basic agent
def get_special_env_plist(
    sp_eih: List[List[str]],
    tag=SPECIAL_ENV_PLIST_TAG,
    max_count=10,
) -> List[TaggedMessage]:
    """(User, EnvAgent, CommonAgent, ContextProvider)

    Get special env plist that precedes `special_env` generation.

    ```
    On EnvAgent (After a special env is available):
    [
        ...,
        ({"role": "user", "content": "..."}, "get_basic_env_plist"),
        > ({"role": "user", "content": "..."}, "get_special_env_plist"),
        ({"role": "user", "content": "..."}, "get_special_env_code_getter_gen_plist"),
        ...,
    ]

    On CommonAgent :
    [
        ...,
        ({"role": "user", "content": "..."}, "get_basic_env_plist"),
        > ({"role": "user", "content": "..."}, "get_special_env_plist"),
        ({"role": "user", "content": "..."}, "get_strat_gen_plist"),
        ...,
        ({"role": "user", "content": "..."}, "get_basic_env_plist"),
        > ({"role": "user", "content": "..."}, "get_special_env_plist"),
        ({"role": "user", "content": "..."}, "get_strat_code_gen_plist"),
        ...,
    ]
    ```

    Args:
        sp_eih (List[List[str]]): Special env info from docker execution.
        tag (str, optional): Tag to identify the only message in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    if len(sp_eih) > 1:
        inner_env_infos = []

        for env_infos in sp_eih:
            for env_info in env_infos:
                inner_env_infos.append(
                    SINGULAR_BASIC_ENV_INFO_INCLUSION_TEMPLATE.format(
                        special_env_info=env_info
                    )
                )

        outer_env_infos = PLURAL_BASIC_ENV_INFO_INCLUSION_TEMPLATE.format(
            basic_env_infos="".join(inner_env_infos[-max_count:])
        )
    else:
        outer_env_infos = SINGULAR_BASIC_ENV_INFO_INCLUSION_TEMPLATE.format(
            basic_env_info=sp_eih[0]
        )

    return [
        ({"role": "user", "content": outer_env_infos}, tag),
    ]


STRAT_REQ_TEMPLATE = """
Given the previous context, generate a few specific potential strategies as an AI agent to achieve freeing up some disk space for the directory `{in_con_path}`. 
Please generate the task in JSON list exactly formatted like {{"list": ["strategy1", "strategy2", ... "strategyN"]}}. 
""".strip()
STRAT_REQ_PLIST_TAG = "get_strat_req_plist"


def get_strat_req_plist(
    in_con_path: str | Path,
    tag=STRAT_REQ_PLIST_TAG,
) -> List[TaggedMessage]:
    """(User, CommonAgent, GenerationRequest, ListOutput)

    Get strat gen plist for strat generation.

    ```
    On CommonAgent :
    [
        ...,
        ({"role": "user", "content": "..."}, "get_special_env_plist"),
        > ({"role": "user", "content": "..."}, "get_strat_gen_plist"),
        ({"role": "assistant", "content": "..."}, "gen_list"),
        ...,
    ]
    ```

    Args:
        special_env_info (str): Special env info from docker execution.
        strat_template (str, optional): Prompt template.
        tag (str, optional): Tag to identify the only message in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    strat_prompt = STRAT_REQ_TEMPLATE.format(
        in_con_path=in_con_path  #
    )

    return [
        ({"role": "user", "content": strat_prompt}, tag),
    ]


SPECIAL_EGC_REQ_TEMPLATE = """
Given the previous context, generate python code for obtaining special environment description that is going to assists you with tasks defined in the system prompt.
You are working in the directory of {in_con_path}.
Be original, unique, and create other information that has yet existed in the previous environment info.
Also use default python library and nothing else.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
The code are expected to print out string to stdout containing environment information.
If there is another code you have previously generated, be different and unique to the previous one.
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`for without installing another new library.
""".strip()
SPECIAL_EGC_REQ_PLIST_TAG = "get_special_egc_req_plist"


# Used by env agent
def get_special_egc_req_plist(
    in_con_path: str | Path,
    tag=SPECIAL_EGC_REQ_PLIST_TAG,
) -> List[TaggedMessage]:
    """(User, EnvAgent, GenerationRequest, CodeOutput)

    Get a plist for special EGC (Environment Getter Code) generation request.

    ```
    On EnvAgent :
    [
        ...,
        ({"role": "user", "content": "..."}, "get_special_env_plist"),
        ({"role": "user", "content": "..."}, "get_special_env_code_getter_gen_plist"),
        ({"role": "assistant", "content": "..."}, "gen_code"),
        ...,
    ]
    ```

    Args:
        in_con_path (str | Path): In container path.
        template (str, optional): Prompt template.
        tag (str, optional): Tag to identify the only message in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    special_env_code_getter_prompt = SPECIAL_EGC_REQ_TEMPLATE.format(
        in_con_path=in_con_path
    )

    return [
        ({"role": "user", "content": special_env_code_getter_prompt}, tag),
    ]


STRAT_CODE_REQ_TEMPLATE = """
Given the previous context, generate python code for the task \"{task_description}\".
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code :\n
""".strip()

STRAT_CODE_REQ_TAG = "get_strat_code_req_plist"


# Used by basic agent
def get_strat_code_gen_plist(
    task_description: str,
    tag=STRAT_CODE_REQ_TAG,
) -> List[TaggedMessage]:
    """(User, CommonAgent, GenerationRequest, CodeOutput)

    Get a plist for strat code generation.

    ```
    On EnvAgent :
    [
        ...,
        ({"role": "user", "content": "..."}, "get_special_env_plist"),
        > ({"role": "user", "content": "..."}, "get_strat_code_gen_plist"),
        ({"role": "assistant", "content": "..."}, "gen_code"),
        ...,
    ]
    ```

    Args:
        in_con_path (str | Path): In container path.
        template (str, optional): Prompt template.
        tag (str, optional): Tag to identify the only message in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    code_prompt = STRAT_CODE_REQ_TEMPLATE.format(
        task_description=task_description  #
    )

    return [
        ({"role": "user", "content": code_prompt}, tag),
    ]


REGEN_TEMPLATE = """
Output of the code above is after running it on {run_context} is:
{error_context}
Please improve upon the code you have written by keeping in mind the error output above where task is {task_description}.
Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}. 
DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
Code :\n
""".strip()

USER_CODE_REGEN_TAG = "get_code_regen_plist"


# Used by basic and env agent
def get_code_regen_plist(
    task_description: str,
    error_context: str,
    run_context: str,
    user_tag=USER_CODE_REGEN_TAG,
) -> List[TaggedMessage]:
    """(User, CommonAgent, EnvAgent, GenerationRequest, CodeOutput)

    Get a plist for code regeneration.

    ```
    On CommonAgent and CommonAgent :
    [
        ...,
        ({"role": "user", "content": "..."}, "*REQ*"),
        > ({"role": "assistant", "content": "..."}, ASSISTANT_REGEN_TAG),
        > ({"role": "user", "content": "..."}, USER_REGEN_TAG),
        ({"role": "assistant", "content": "..."}, "gen_code"),
        ...,
    ]
    ```

    Args:
        task_description (str): Task/strat context for regen.
        asssitant_raw_response (str): Previously generated response by assistant.
        error_context (str): Error generated by previous code.
        run_context (str): Execution context of previous code
        user_tag (str, optional): Tag to identify the regen request in plist.
        assi_tag (str, optional): Tag to identify assistant's failed code in plist.

    Returns:
        List[TaggedMessage]: TaggedMessages to append to caller's chat history.
    """
    regen_prompts = REGEN_TEMPLATE.format(
        task_description=task_description,
        error_context=error_context,
        run_context=run_context,
    )

    return [
        ({"role": "user", "content": regen_prompts}, user_tag),
    ]


# IMPROVE_TEMPLATE = """
# Results of the code above is after running it on {run_context} are:
# {newest_env_info}
# With the difference of (Previous - Current):
# {env_info_diff}
# Please improve upon the code you have written by keeping in mind the error output above where task is {task_description}.
# Please generate the code in JSON format exactly formatted like {{"code": "import ... "}}.
# DO NOT GENERATE ANY COMMENTS, you are expected to generate code that can be run using python's `eval()`.
# Code:\n
# """.strip()


# # Used by basic and env agent
# def get_improve_plist_oai(
#     prev_code: str,
#     newest_env_info: EnvironmentInfo,
#     diff_env: EnvironmentInfo,
#     task_description: str,
#     run_context: str,
#     improve_template=IMPROVE_TEMPLATE,
# ) -> List[TaggedMessage]:
#     improve_prompt = improve_template.format(
#         run_context=run_context,
#         newest_env_info=newest_env_info.model_dump_json(indent=4),
#         env_info_diff=diff_env.model_dump_json(indent=4),
#         task_description=task_description,
#     )

#     return [
#         ({"role": "assistant", "content": prev_code}, "successful_code"),
#         ({"role": "user", "content": improve_prompt}, "improve_request"),
#     ]
