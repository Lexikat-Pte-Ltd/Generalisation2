from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import List, Literal, cast, get_args

import docker
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger

from src.code import is_valid_code_ast, is_valid_code_compiler
from src.container import (
    run_code_in_con,
    safe_detect_env,
)
from src.data import EnvironmentInfo
from src.helper import format_tch
from src.prep import (
    get_code_regen_plist,
    get_strat_code_gen_plist,
)
from src.gen import create_genner, gen_code
from src.agent import EnvAgent, CommonAgent

load_dotenv()

EA_MAX_ATTEMPTS = 5
EA_MAX_SP_EGC = 2
CA_MAX_ATTEMPTS = 3

CONTAINER_ID = "learn-compose-main-1"
BACKEND = cast(Literal["oai", "deepseek"], os.getenv("BACKEND", "oai"))
OAI_API_KEY = os.getenv("OPENAI_API_KEY")

if BACKEND == "oai":
    assert OAI_API_KEY is not None

oai_client = OpenAI(api_key=OAI_API_KEY)
docker_client = docker.from_env()
genner = create_genner(BACKEND, oai_client=oai_client)


def main():
    save_folder = Path() / "data"
    # In container path for work path purposes
    in_con_path = Path("/")

    # Get basic initial env info
    logger.info("Initializing basic environment info.")
    initial_basic_env_info: EnvironmentInfo = safe_detect_env(
        docker_client, CONTAINER_ID
    )

    # Initiate env agent
    logger.info("EA - Initializing environment agent.")
    ea = EnvAgent(
        initial_basic_env_info=initial_basic_env_info, in_con_path=in_con_path
    )

    # Generate 2 special env getter code
    logger.info(f"EA - Generating {EA_MAX_SP_EGC} special environment getter code.")
    new_tch, new_sp_egc_s = ea.gen_multi_sp_egc(
        count=EA_MAX_SP_EGC,
        in_con_path=in_con_path,
        genner=genner,
        docker_client=docker_client,
        testing_container_id=CONTAINER_ID,
        max_attempts=EA_MAX_ATTEMPTS,
    )
    ea.tagged_chat_history.extend(new_tch)
    ea.sp_env_info_getter_codes.extend(new_sp_egc_s)

    # Execute all env getting codes that has obtained
    logger.info(f"EA - Executing all SP EGC code on container {CONTAINER_ID} ...")
    initial_special_env_infos = ea.execute_sp_env_infos(
        docker_client=docker_client, container_id=CONTAINER_ID
    )

    # Update env_agent's state
    logger.info("EA - Updating environment agent's SP EIH state...")
    ea.sp_env_info_getter_codes.extend(initial_special_env_infos)

    # Intiiate common agent
    common_agent = CommonAgent(
        bs_env_info_history=ea.bs_env_info_history,
        sp_env_info_history=ea.sp_env_info_history,
        in_con_path=in_con_path,
    )

    # Generate strats based on the new env infos
    strats, raw_strats = common_agent.gen_strats(genner)

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
    for i, strat in enumerate(strats):
        logger.info(f"CA - {i}-th strat - Code loop started on strat {strat}...")
        # Copy the agent
        cur_ca = deepcopy(common_agent)

        # Get fresh env info
        loop_bs_env_info = safe_detect_env(docker_client, CONTAINER_ID)

        # Update env agent
        ea.bs_env_info_history.append(loop_bs_env_info)
        ea.sp_env_info_history.append(
            ea.execute_sp_env_infos(
                docker_client=docker_client, container_id=CONTAINER_ID
            )
        )

        # Update the env info state of the common agent with the 2 fresh env infos
        changes = cur_ca.update_env_info_state(
            ea.bs_env_info_history, ea.sp_env_info_history
        )
        logger.info(
            f"CA - {i}-th strat - Number of updates on common agent env info's state {changes}"
        )
        # Save data before proceeding
        cur_ca.save_data(
            space_freed=-1,  # No space freed but can be useful maybe
            strat=strat,
            folder=save_folder,
        )
        logger.info(
            f"CA - {i}-th strat - Saved data @ {len(cur_ca.tagged_chat_history)} on {save_folder}."
        )

        # Prepare the common agent chat history with prep prompts
        cur_ca.tagged_chat_history.extend(
            get_strat_code_gen_plist(task_description=strat)
        )

        # Gen some codes
        attempt = 0
        while attempt < CA_MAX_ATTEMPTS:
            logger.debug(
                (
                    f"CA - {i}-th code - {attempt}-th attempt - ",
                    f"CommonAgent's in loop tagged chat history - \n{
                        format_tch(cur_ca.tagged_chat_history)}",
                )
            )

            code, raw_response = gen_code(genner, common_agent.chat_history)
            cur_ca.tagged_chat_history.extend(
                [({"role": "assistant", "content": raw_response}, "gen_sp_egc")]
            )

            ast_valid, ast_error = is_valid_code_ast(code)
            if not ast_valid:
                logger.error(
                    f"CA - {i}-th strat - {attempt + 1}-th attempt - "
                    f"AST error \n{ast_error}"
                )
                logger.debug(f"CA - Code is \n{code}")

                cur_ca.tagged_chat_history.append(
                    (
                        {"role": "assistant", "content": raw_response},
                        "strat_codegen(ast_fail)",
                    )
                )
                cur_ca.tagged_chat_history.extend(
                    get_code_regen_plist(
                        task_description=f"Generate code to perform {strat}",
                        error_context=ast_error,
                        run_context="a Python AST compiler",
                    )
                )

                attempt += 1
                continue

            compile_valid, compiler_error = is_valid_code_compiler(code)

            if not compile_valid:
                logger.error(
                    f"CA - {i}-th strat - {attempt + 1}-th attempt - "
                    f"Native `compile` error \n{compiler_error}"
                )
                logger.debug(f"CA - Code is \n{code}")

                cur_ca.tagged_chat_history.append(
                    (
                        {"role": "assistant", "content": raw_response},
                        "strat_codegen(compile_fail)",
                    )
                )
                cur_ca.tagged_chat_history.extend(
                    get_code_regen_plist(
                        task_description=f"Generate code to perform {strat}",
                        error_context=ast_error,
                        run_context="a Python native compiler",
                    )
                )

                attempt += 1
                continue

            exit_code, execution_output = run_code_in_con(
                docker_client, CONTAINER_ID, code
            )
            if exit_code != 0:
                logger.error(
                    f"CA - {i}-th strat - {attempt + 1}-th attempt - "
                    f"In container error \n{execution_output}"
                )
                logger.debug(f"CA - Code is \n{code}")

                cur_ca.tagged_chat_history.append(
                    (
                        {"role": "assistant", "content": raw_response},
                        "strat_codegen(container_fail)",
                    )
                )
                cur_ca.tagged_chat_history.extend(
                    get_code_regen_plist(
                        task_description=f"Generate code to perform {strat}",
                        error_context=execution_output,
                        run_context="a Docker container",
                    )
                )

                attempt += 1
                continue

            fresh_basic_env_info = safe_detect_env(docker_client, CONTAINER_ID)
            storage_changes, files_deleted = loop_bs_env_info.total_files_deleted(
                fresh_basic_env_info
            )

            if not files_deleted:
                logger.error(
                    f"CA - {i}-th strat - {attempt + 1}-th attempt - "
                    f"No spaces are freed \n{execution_output}"
                )
                logger.debug(f"CA - Code is \n{code}")

                cur_ca.tagged_chat_history.append(
                    (
                        {"role": "assistant", "content": raw_response},
                        "strat_codegen(deletion_fail)",
                    )
                )
                cur_ca.tagged_chat_history.extend(
                    get_code_regen_plist(
                        task_description=f"Generate code to perform {strat}",
                        error_context="No files are being freed or deleted",
                        run_context="a Docker container",
                    )
                )

                attempt += 1
                continue

        if attempt >= CA_MAX_ATTEMPTS:
            logger.info(f"CA - {i}-th strat - Codegen loop ended up in failure")
        else:
            logger.info(f"CA - {i}-th strat - Codegen loop ended up in success")
            cur_ca.tagged_chat_history.append(
                (
                    {"role": "assistant", "content": raw_response},
                    "strat_codegen(success)",
                )
            )

        logger.debug(f"CA - {i}-th strat - Code is \n{code}")

        # Lets save it
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d%m%y_%H%M")

        cur_ca.save_data(
            space_freed=storage_changes,  # No space freed but can be useful maybe
            strat=strat,
            folder=save_folder,
        )
        ea.save_data(folder=save_folder)

        logger.info(
            "Code loop is done for the strat {strat} on {attempt}-th iteration."
        )
        logger.info("Continuing to next strat if exists...")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    if not args.debug:
        logger.remove()

    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}"
    )

    logger.add(
        sys.stderr,
        backtrace=args.debug,
        diagnose=args.debug,
        format=logger_format,
        level="DEBUG" if args.debug else "INFO",
    )

    main()
