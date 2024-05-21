import json
from pathlib import Path
from pprint import pformat
from typing import Dict, List

from docker import DockerClient
from loguru import logger
from openai import OpenAI

from src.code import (
    is_valid_code_ast,
    is_valid_code_compiler,
    process_response_code,
)
from src.container import run_code_in_con, safe_detect_env
from src.prompt import (
    gen_skill_code_openai,
    prepare_regenerate_prompts_openai,
    prepare_improve_prompt_openai,
)


def save_skill_openai(
    task_description: str,
    prompts: List[Dict[str, str]],
    failures: List[str],
    folder_name: str | Path,
    is_successful: bool,
) -> str:
    prompts_ = prompts if is_successful else prompts[:2] + prompts[-1:]
    data = {
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


def run_loop(
    docker_client: DockerClient,
    openai_client: OpenAI,
    container_id: str,
    task_description: str,
    initial_code: str,
    initial_prompts: List[Dict[str, str]],
    skill_folder: str | Path,
    max_loop=7,
) -> bool:
    cur_prompts = initial_prompts.copy()
    cur_code = initial_code
    cur_failures = list()

    for i in range(max_loop):
        cur_env = safe_detect_env(docker_client, container_id)
        _, ast_error = is_valid_code_ast(cur_code)

        if ast_error:
            logger.error(
                f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - AST Parse error - {ast_error}"
            )
            logger.debug(f"{cur_env.model_dump_json()}")
            logger.debug(f"{cur_code}")

            cur_prompts = cur_prompts + prepare_regenerate_prompts_openai(
                task_description=task_description,
                prev_code=cur_code,
                run_context="`AST.parse()`",
                error_context=ast_error,
            )

            cur_code = gen_skill_code_openai(openai_client, cur_prompts)
            cur_code = process_response_code(cur_code)
            cur_failures.append("ast_parse")

            continue

        _, compiler_error = is_valid_code_compiler(cur_code)

        if compiler_error:
            logger.error(
                f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Native `compile` error - {compiler_error}"
            )
            logger.debug(f"{cur_env.model_dump_json()}")
            logger.debug(f"{cur_code}")

            cur_prompts = cur_prompts + prepare_regenerate_prompts_openai(
                task_description=task_description,
                prev_code=cur_code,
                run_context="`compile(code, '<string>', 'exec')`",
                error_context=ast_error,
            )

            cur_code = gen_skill_code_openai(openai_client, cur_prompts)
            cur_code = process_response_code(cur_code)
            cur_failures.append("compiler_function")

            continue

        _, container_error = run_code_in_con(
            docker_client,
            container_id,
            cur_code,
        )

        if container_error:
            logger.error(
                f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - In container error - {container_error}"
            )
            logger.debug(f"{cur_env.model_dump_json()}")
            logger.debug(f"{cur_code}")

            cur_prompts = cur_prompts + prepare_regenerate_prompts_openai(
                task_description=task_description,
                prev_code=cur_code,
                run_context="`container.exec_run(cmd='python -c {{code}}')`",
                error_context=ast_error,
            )

            cur_code = gen_skill_code_openai(openai_client, cur_prompts)
            cur_code = process_response_code(cur_code)
            cur_failures.append("container_run")

            continue

        newest_env_info = safe_detect_env(docker_client, container_id)

        if False:
            logger.error(
                f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Doesnt really empty space."
            )
            logger.debug(f"{cur_env.model_dump_json()}")
            logger.debug(f"{cur_code}")

            cur_prompts = cur_prompts + prepare_improve_prompt_openai(
                prev_code=cur_code,
                newest_env_info=newest_env_info,
                diff_env=cur_env.diff(newest_env_info),
                task_description=task_description,
                run_context="`container.exec_run(cmd='python -c {{code}}')`",
            )
            logger.debug(pformat(cur_prompts))

            cur_code = gen_skill_code_openai(openai_client, cur_prompts)
            cur_code = process_response_code(cur_code)
            cur_failures.append("runs_but_not_deleting")

            continue

        cur_failures.append("not_a_failure")
        cur_prompts.append({"role": "assistant", "content": cur_code})

        # Save to "successful" and return early
        skill_path = save_skill_openai(
            task_description=task_description,
            prompts=cur_prompts,
            failures=cur_failures,
            folder_name=skill_folder,
            is_successful=True,
        )

        logger.info(
            f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Learn loop succeed, dumping skill in {skill_path}"
        )
        logger.debug(task_description)
        logger.debug(cur_prompts[-1])
        logger.debug(cur_failures)

        return True

    # Save to "failure" and return
    skill_path = save_skill_openai(
        task_description=task_description,
        prompts=cur_prompts,
        failures=cur_failures,
        folder_name=skill_folder,
        is_successful=True,
    )
    logger.info(
        f"run_loop {i}-th {docker_client.api.base_url} {container_id} {task_description} - Learn loop ends in failure, dumping skill in {skill_path}"
    )
    logger.debug(task_description)
    logger.debug(cur_prompts)
    logger.debug(cur_failures)

    return False
