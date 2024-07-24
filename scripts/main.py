from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
from typing import List, Literal, cast, get_args

import docker
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

from src.code import is_valid_code_ast, is_valid_code_compiler
from src.container import (
    run_code_in_con,
    safe_detect_env,
    safe_folderlist_bfs,
)
from src.data import EnvironmentInfo
from src.prep import (
    get_code_regen_plist,
    get_special_egc_req_plist,
    get_strat_code_gen_plist,
)
from src.gen import create_genner, gen_code
from src.agent import EnvAgent, CommonAgent

load_dotenv()

CONTAINER_ID = "learn-compose-main-1"
BACKEND = cast(Literal["oai", "deepseek"], os.getenv("BACKEND") or "oai")
assert BACKEND == "oai"
OAI_API_KEY = os.getenv("OPENAI_API_KEY")

if BACKEND == "oai":
    assert OAI_API_KEY is not None

oai_client = OpenAI(api_key=OAI_API_KEY)
docker_client = docker.from_env()

genner = create_genner(BACKEND, oai_client=oai_client)


def main():
    # Intiate both docker and openai clients

    # In container path for work path purposes
    in_con_path = Path("/")
    # In con folders for adding context to environment data getting
    in_con_folders = safe_folderlist_bfs(
        client=docker_client, container_id=CONTAINER_ID, path=in_con_path
    )
    # Filtering so that context size doesnt grow too much
    smol_in_con_folders = in_con_folders[:25] + ["..."] + in_con_folders[-1:]

    # Get basic initial env info
    logger.info("Initializing basic environment info.")
    initial_basic_env_info: EnvironmentInfo = safe_detect_env(
        docker_client, CONTAINER_ID
    )

    # Initiate env agent
    logger.info("Initializing environment agent.")
    env_agent = EnvAgent(
        initial_basic_env_info=initial_basic_env_info, in_con_path=in_con_path
    )

    # Generate 2 special env getter code
    logger.info("Generating 2 special environment getter code...")
    for i in range(2):
        env_agent.tagged_chat_history.extend(get_special_egc_req_plist(in_con_path))

        env_getter_code, succeed = env_agent.gen_sp_egc(
            genner=genner,
            docker_client=docker_client,
            testing_container_id=CONTAINER_ID,
        )

        # If special environment getters generation process failed
        if not succeed:
            logger.error("Failed generating env getter code")
            raise Exception("Somehow `gen_special_env_getters` failed.")

        # Update (append) env_agent's state
        logger.info(f"Appending env_agent with new env getter code @ {i} th iteration.")
        env_agent.append_new_code(env_getter_code)

    # Execute all env getting codes that has obtained
    logger.info(
        f"Executing all special environment info getter code on container {CONTAINER_ID} ..."
    )
    initial_special_env_infos = env_agent.execute_sp_env_infos(
        docker_client=docker_client, container_id=CONTAINER_ID
    )

    # Update env_agent's state
    logger.info("Updating environment agent's state...")
    env_agent.sp_env_info_history.extend(initial_special_env_infos)

    # Intiiate common agent
    common_agent = CommonAgent(
        basic_env_info_history=env_agent.bs_env_info_history,
        sp_env_info_history=env_agent.sp_env_info_history,
        in_con_path=in_con_path,
    )

    # Generate strats based on the new env infos
    strats, raw_strats = common_agent.gen_strats(genner)
    common_agent.debug_log()

    env_agent.debug_log_bs_eih()
    env_agent.debug_log_sp_eih()
    env_agent.debug_log_tch()

    # Strat deletion strats :
    # 1. Regex or using BERT to remove similar strats that had been worked with before
    # 2. Start from last to beginning
    # 3. Prompt engineering to tell it to avoid to generate previously generated strats
    # 4. Soft prompting

    # For every strat, we
    # 1. Copy the common agent
    # 2. Refresh basic env info and special env info
    # 3. Update the state of the common agent
    # 4. Generate codes to free up spaces
    # 5. Save the code whether if it succeed or not
    for strat in strats:
        logger.info("Code loop started on strat {strat}...")
        # Copy the agent
        strat_focused_common_agent = deepcopy(common_agent)

        # Get fresh env info
        starting_basic_env_info = safe_detect_env(docker_client, CONTAINER_ID)
        starting_special_env_infos: List[str] = env_agent.execute_sp_env_infos(
            docker_client=docker_client, container_id=CONTAINER_ID
        )

        # Update the env info state of the common agent with the 2 fresh env infos
        changes = common_agent.update_env_info_state(
            starting_basic_env_info, starting_special_env_infos
        )
        logger.info("Number of updates on common agent env info's state {changes}")

        # Prepare the common agent chat history with prep prompts
        common_agent.tagged_chat_history += get_strat_code_gen_plist(
            task_description=strat
        )

        # Gen some codes
        for attempt in range(3):
            code, raw_response = gen_code(genner, common_agent.chat_history)
            common_agent.tagged_chat_history += [
                ({"role": "assistant", "content": raw_response}, "gen_sp_egc")
            ]

            ast_valid, ast_error = is_valid_code_ast(code)
            if not ast_valid:
                logger.error(
                    f"AST error - Attempt number {attempt + 1} - \n{ast_error}"
                )
                common_agent.tagged_chat_history += get_code_regen_plist(
                    task_description=f"Generate code to perform {strat}",
                    error_context=ast_error,
                    run_context="a Python AST compiler",
                )
                continue

            compile_valid, compiler_error = is_valid_code_compiler(code)

            if not compile_valid:
                logger.error(
                    f"Native `compile` error - Attempt number {attempt + 1} - \n{compiler_error}"
                )
                common_agent.tagged_chat_history += get_code_regen_plist(
                    task_description=f"Generate code to perform {strat}",
                    error_context=ast_error,
                    run_context="a Python native compiler",
                )
                continue

            exit_code, execution_output = run_code_in_con(
                docker_client, CONTAINER_ID, code
            )
            if exit_code != 0:
                logger.error(
                    f"In container error - Attempt number {attempt + 1} - \n{execution_output}"
                )
                common_agent.tagged_chat_history += get_code_regen_plist(
                    task_description=f"Generate code to perform {strat}",
                    error_context=execution_output,
                    run_context="a Docker container",
                )
                continue

            fresh_basic_env_info = safe_detect_env(docker_client, CONTAINER_ID)
            files_are_deleted = starting_basic_env_info.files_are_deleted(
                fresh_basic_env_info
            )

            if not files_are_deleted:
                logger.error(
                    f"No deleted files are detected- Attempt number {attempt + 1} - \n{execution_output}"
                )
                common_agent.tagged_chat_history += get_code_regen_plist(
                    task_description=f"Generate code to perform {strat}",
                    error_context="No files are being freed or deleted",
                    run_context="a Docker container",
                )
                continue

            # Files are deleted, lets save it
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%d%m%y_%H%M")

            common_agent.save_tagged_chat_history_to_json(
                identifier=formatted_datetime, folder="data/learned_skills/"
            )
            env_agent.save_tagged_chat_history_to_json(
                identifier=formatted_datetime, folder="data/learned_skills/"
            )

            logger.info(
                "Code loop is done for the strat {strat} on {attempt}-th iteration."
            )
            logger.info("Continuing to next strat if exists...")


if __name__ == "__main__":
    main()
