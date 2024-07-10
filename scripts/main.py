from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path

import docker
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

from src.code import is_valid_code_ast, is_valid_code_compiler
from src.container import (
    run_code_in_con,
    safe_detect_env,
    safe_filelist,
    safe_folderlist_bfs,
)
from src.prep import (
    prep_regen_plist_oai,
    prep_special_env_codegen_plist_oai,
    prep_strat_codegen_plist_oai,
    prep_stratgen_plist_oai,
    prep_special_env_codegen_plist_oai,
)
from src.gen import gen_code_oai
from src.looper import run_strategy
from src.helper import to_normal_plist
from src.agent import EnvAgent, CommonAgent

load_dotenv()

if __name__ == "__main__":
    container_id = "learn-compose-main-1"

    # Intiate both docker and openai clients
    docker_client = docker.from_env()
    oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Out container path for saving purposes
    out_con_path = Path("./data/controller/")
    # In container path for work path purposes
    in_con_path = Path("/")
    # In con folders for adding context to environment data getting
    in_con_folders = safe_folderlist_bfs(
        client=docker_client, container_id=container_id, path=in_con_path
    )
    # Filtering so that context size doesnt grow too much
    smol_in_con_folders = in_con_folders[:25] + ["..."] + in_con_folders[-1:]

    # Get basic initial env info
    initial_basic_env_info = safe_detect_env(docker_client, container_id)

    # Initiate env agent
    env_agent = EnvAgent(
        initial_basic_env_info=initial_basic_env_info, in_con_path=in_con_path
    )

    # Generate 2 special env getter code
    for i in range(2):
        env_agent.tagged_chat_history += prep_special_env_codegen_plist_oai(in_con_path)

        env_getter_code, succeed = env_agent.gen_special_env_code(
            oai_client=oai_client,
            docker_client=docker_client,
            testing_container_id=container_id,
        )

        # If special environment getters generation process failed
        if not succeed:
            logger.error("Failed generating env getter code")
            raise Exception("Somehow `gen_special_env_getters` failed.")

        # Update (append) env_agent's state
        env_agent.append_new_code(env_getter_code)

    # Execute all env getting codes that has obtained
    initial_special_env_infos = env_agent.execute_special_env_infos(
        docker_client=docker_client, container_id=container_id
    )

    # Update env_agent's state
    env_agent.cur_sp_env_infos = initial_special_env_infos
    env_agent.log_tagged_chat_history()

    # Intiiate common agent
    common_agent = CommonAgent(
        initial_basic_env_info=env_agent.cur_basic_env_info,
        initial_special_env_infos=env_agent.cur_sp_env_infos,
        in_con_path=in_con_path,
    )

    # Generate strats based on the new env infos
    strats = common_agent.gen_strats(oai_client)
    common_agent.log_tagged_chat_history()

    # For every strat, we
    # 1. Copy the common agent
    # 2. Refresh basic env info and special env info
    # 3. Update the state of the common agent
    # 4. Generate codes to free up spaces
    # 5. Save the code whether if it succeed or not
    for strat in strats:
        # Copy the agent
        strat_focused_common_agent = deepcopy(common_agent)

        # Get fresh env info
        starting_basic_env_info = safe_detect_env(docker_client, container_id)
        starting_special_env_infos = env_agent.execute_special_env_infos(
            docker_client=docker_client, container_id=container_id
        )

        # Update the env info state of the common agent with the 2 fresh env infos
        changes = common_agent.update_env_info_state(
            starting_basic_env_info, starting_special_env_infos
        )
        logger.info("Number of updates on common agent env info's state {changes}")

        # Prepare the common agent chat history with prep prompts
        common_agent.tagged_chat_history += prep_strat_codegen_plist_oai(
            task_description=strat
        )
        # Gen some codes
        for attempt in range(3):
            code: str = gen_code_oai(oai_client, common_agent.chat_history)

            ast_valid, ast_error = is_valid_code_ast(code)
            if not ast_valid:
                logger.error(
                    f"AST error - Attempt number {attempt + 1} - \n{ast_error}"
                )
                common_agent.tagged_chat_history += prep_regen_plist_oai(
                    task_description=f"Generate code to perform {strat}",
                    prev_code=code,
                    error_context=ast_error,
                    run_context="a Python AST compiler",
                )
                continue
            compile_valid, compiler_error = is_valid_code_compiler(code)
            if not compile_valid:
                logger.error(
                    f"Native `compile` error - Attempt number {attempt + 1} - \n{compiler_error}"
                )
                common_agent.tagged_chat_history += prep_regen_plist_oai(
                    task_description=f"Generate code to perform {strat}",
                    prev_code=code,
                    error_context=ast_error,
                    run_context="a Python native compiler",
                )
                continue

            exit_code, execution_output = run_code_in_con(
                docker_client, container_id, code
            )
            if exit_code != 0:
                logger.error(
                    f"In container error - Attempt number {attempt + 1} - \n{execution_output}"
                )
                common_agent.tagged_chat_history += prep_regen_plist_oai(
                    task_description=f"Generate code to perform {strat}",
                    prev_code=code,
                    error_context=execution_output,
                    run_context="a Docker container",
                )
                continue

            fresh_basic_env_info = safe_detect_env(docker_client, container_id)
            files_are_deleted = starting_basic_env_info.files_are_deleted(
                fresh_basic_env_info
            )

            if not files_are_deleted:
                logger.error(
                    f"No deleted files are detected- Attempt number {attempt + 1} - \n{execution_output}"
                )
                common_agent.tagged_chat_history += prep_regen_plist_oai(
                    task_description=f"Generate code to perform {strat}",
                    prev_code=code,
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