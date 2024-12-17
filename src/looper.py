import json
from pathlib import Path
from typing import Dict, List, Any

from docker import DockerClient
from openai import OpenAI

from src.types import TaggedPList


def save_tagged_history_openai(
    task_description: str,
    tagged_plist: TaggedPList,
    failures: List[str],
    folder_name: str | Path, is_successful: bool,
) -> str:
    prompts_ = (
        tagged_plist if is_successful else tagged_plist.messages[:2] + tagged_plist.messages[-1:]
    )
    data: Dict[str, Any] = {
        "task_name": task_description,
        "prompt_history": prompts_,
        "failures": failures,
    }

    save_folder = "learned_skills" if is_successful else "failed_skills"

    filename = (
        "_".join(task_description.lower().split(" ")).replace("/", "").replace("\\", "")
        + ".json"
    )
    filepath = Path(folder_name) / save_folder / filename

    json_data = json.dumps(data, indent=4)
    filepath.write_text(json_data, encoding="utf-8")

    return str(filepath)


def run_strategy(
    docker_client: DockerClient,
    openai_client: OpenAI,
    container_id: str,
    task_description: str,
    initial_code: str,
    initial_prompts: TaggedPList,
    skill_folder: str | Path,
    max_loop=7,
) -> bool:
    return True


#     cur_plist_hist = initial_prompts.copy()
#     cur_code = initial_code
#     cur_failures = list()

#     for i in range(max_loop):
#         cur_env = safe_detect_env(docker_client, container_id)
#         _, ast_error = is_valid_code_ast(cur_code)

#         if ast_error:
#             logger.error(
#                 f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - AST Parse error - {ast_error}"
#             )
#             logger.debug(f"{cur_env.model_dump_json()}")
#             logger.debug(f"{cur_code}")

#             regen_plist, plist_type =

#             cur_plist_hist = cur_plist_hist + prep_regen_plist_oai(
#                 task_description=task_description,
#                 prev_code=cur_code,
#                 run_context="`AST.parse()`",
#                 error_context=ast_error,
#             )

#             cur_code = gen_skill_code_openai(openai_client, cur_plist_hist)
#             cur_code = process_response_code(cur_code)
#             cur_failures.append("ast_parse")

#             continue

#         _, compiler_error = is_valid_code_compiler(cur_code)

#         if compiler_error:
#             logger.error(
#                 f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Native `compile` error - {compiler_error}"
#             )
#             logger.debug(f"{cur_env.model_dump_json()}")
#             logger.debug(f"{cur_code}")

#             cur_plist_hist = cur_plist_hist + prep_regen_plist_oai(
#                 task_description=task_description,
#                 prev_code=cur_code,
#                 run_context="`compile(code, '<string>', 'exec')`",
#                 error_context=ast_error,
#             )

#             cur_code = gen_skill_code_openai(openai_client, cur_plist_hist)
#             cur_code = process_response_code(cur_code)
#             cur_failures.append("compiler_function")

#             continue

#         _, container_error = run_code_in_con(
#             docker_client,
#             container_id,
#             cur_code,
#         )

#         if container_error:
#             logger.error(
#                 f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - In container error - {container_error}"
#             )
#             logger.debug(f"{cur_env.model_dump_json()}")
#             logger.debug(f"{cur_code}")

#             cur_plist_hist = cur_plist_hist + prep_regen_plist_oai(
#                 task_description=task_description,
#                 prev_code=cur_code,
#                 run_context="`container.exec_run(cmd='python -c {{code}}')`",
#                 error_context=ast_error,
#             )

#             cur_code = gen_skill_code_openai(openai_client, cur_plist_hist)
#             cur_code = process_response_code(cur_code)
#             cur_failures.append("container_run")

#             continue

#         newest_env_info = safe_detect_env(docker_client, container_id)

#         if False:
#             logger.error(
#                 f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Doesnt really empty space."
#             )
#             logger.debug(f"{cur_env.model_dump_json()}")
#             logger.debug(f"{cur_code}")

#             cur_plist_hist = cur_plist_hist + prep_improve_plist_oai(
#                 prev_code=cur_code,
#                 newest_env_info=newest_env_info,
#                 diff_env=cur_env.diff(newest_env_info),
#                 task_description=task_description,
#                 run_context="`container.exec_run(cmd='python -c {{code}}')`",
#             )
#             logger.debug(pformat(cur_plist_hist))

#             cur_code = gen_skill_code_openai(openai_client, cur_plist_hist)
#             cur_code = process_response_code(cur_code)
#             cur_failures.append("runs_but_not_deleting")

#             continue

#         cur_failures.append("not_a_failure")
#         cur_plist_hist.append({"role": "assistant", "content": cur_code})

#         # Save to "successful" and return early
#         skill_path = save_skill_openai(
#             task_description=task_description,
#             plist=cur_plist_hist,
#             failures=cur_failures,
#             folder_name=skill_folder,
#             is_successful=True,
#         )

#         logger.info(
#             f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Learn loop succeed, dumping skill in {skill_path}"
#         )
#         logger.debug(task_description)
#         logger.debug(cur_plist_hist[-1])
#         logger.debug(cur_failures)

#         return True

#     # Save to "failure" and return
#     skill_path = save_skill_openai(
#         task_description=task_description,
#         plist=cur_plist_hist,
#         failures=cur_failures,
#         folder_name=skill_folder,
#         is_successful=True,
#     )
#     logger.info(
#         f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Learn loop ends in failure, dumping skill in {skill_path}"
#     )
#     logger.debug(task_description)
#     logger.debug(cur_plist_hist)
#     logger.debug(cur_failures)

#     return False
