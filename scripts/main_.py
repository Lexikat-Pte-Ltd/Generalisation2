import os
from pprint import pformat
import subprocess
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from loguru import logger
from loguru._defaults import LOGURU_FORMAT  # type: ignore
from openai import OpenAI

import docker
import docker.errors
import docker.models.containers
from src.agent import save_list_of_agent_data
from src.agent.EnvAgent import EnvAgent
from src.agent.StrategyAgent import StrategyAgent
from src.code import is_valid_code_ast, is_valid_code_compiler
from src.container import (
    run_code_in_con,
    safe_detect_env,
)
from src.data import EnvironmentInfo
from src.genner import Genner, get_genner
from src.helper import (
    format_tch,
    get_code_diff,
    scan_json_files_for_strat,
    timed_input,
)
from src.prep import (
    get_code_regen_plist,
    get_strat_code_req_plist,
)
from src.types import Message, TaggedMessage

os.environ["TZ"] = "Asia/Jakarta"

# Load environment variables
load_dotenv()

# Constants (keeping original names)
EA_MAX_ATTEMPTS = 5
EA_MAX_SP_EGC = 2
CA_MAX_ATTEMPTS = 3

MAIN_CONTAINER_ID = "fedora-learn-compose"
TEST_CONTAINER_ID = "mini-learn-compose"
BACKEND = "qwen"
OAI_API_KEY = os.getenv("OPENAI_API_KEY")
CUR_DATETIME = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def setup_logger(debug: bool) -> None:
    """
    Set up the logger with appropriate configuration.

    Args:
        debug (bool): Whether to enable debug mode.
    """
    os.environ["TZ"] = "Asia/Jakarta"
    log_format = str(LOGURU_FORMAT).replace(
        "{time:YYYY-MM-DD HH:mm:ss.SSS}",
        "{time:YYYY-MM-DD HH:mm:ss.SSS ZZ}",
    )

    logger.remove()
    logger.add(
        sys.stderr,
        backtrace=debug,
        diagnose=debug,
        format=log_format,
        level="DEBUG" if debug else "INFO",
    )
    logger.add(
        f"logs/main_logs/main_{CUR_DATETIME}.log",
        backtrace=debug,
        diagnose=debug,
        format=log_format,
        level="DEBUG" if debug else "INFO",
    )


def log_environment_variables() -> None:
    """Log the environment variables used in the script."""
    log_data = {
        "EA_MAX_ATTEMPTS": EA_MAX_ATTEMPTS,
        "EA_MAX_SP_EGC": EA_MAX_SP_EGC,
        "CA_MAX_ATTEMPTS": CA_MAX_ATTEMPTS,
        "MAIN_CONTAINER_ID": MAIN_CONTAINER_ID,
        "TEST_CONTAINER_ID": TEST_CONTAINER_ID,
        "BACKEND": BACKEND,
        "OAI_API_KEY": OAI_API_KEY,
    }
    logger.info(f"Environment Variables: {log_data}")


def initialize_clients() -> Tuple[OpenAI, docker.DockerClient, Genner, Genner]:
    """
    Initialize the OpenAI client, Docker client, and code generator.

    Returns:
        Tuple[OpenAI, docker.DockerClient, callable]: Initialized clients and generator.
    """
    if BACKEND == "oai":
        assert OAI_API_KEY is not None, "OpenAI API key is required for OAI backend"

    oai_client = OpenAI(api_key=OAI_API_KEY)
    docker_client = docker.from_env()
    genner = get_genner(BACKEND, oai_client=oai_client)
    backup_genner = get_genner("oai", oai_client=oai_client)

    logger.info(
        f"Clients initialized: {oai_client}, {docker_client}, {genner}, {backup_genner}"
    )

    return oai_client, docker_client, genner, backup_genner


def main(
    debug: bool, always_success: bool, main_container_id: str, test_container_id: str
) -> None:
    """
    Main function to run the environment and code generation process.

    Args:
        debug (bool): Whether to run in debug mode.
        always_success (bool): Whether to always consider the process successful.
        main_container_id (str): ID of the main Docker container.
        test_container_id (str): ID of the test Docker container.
    """
    setup_logger(debug)
    log_environment_variables()
    oai_client, docker_client, genner, backup_genner = initialize_clients()

    save_folder = Path() / "data"
    prev_strats = scan_json_files_for_strat(save_folder)
    in_con_path = Path("/")

    log_data = {
        "save_folder": save_folder,
        "prev_strats": prev_strats,
        "in_con_path": in_con_path,
    }
    logger.info(f"Log data: {log_data}")

    timed_input("Enter to continue or wait 3 seconds...")

    # Initialize environment agent
    logger.info("Initializing basic environment info.")

    initial_basic_env_info: EnvironmentInfo = safe_detect_env(
        docker_client, main_container_id
    )

    logger.info("EA - Initializing environment agent.")
    ea = EnvAgent(
        initial_basic_env_info=initial_basic_env_info, in_con_path=in_con_path
    )

    new_tch = ea.get_initial_tch()
    logger.debug(f"EA - New TCH -\n{format_tch(new_tch)}")
    ea.tagged_chat_history.messages.extend(new_tch.messages)

    # Generate special environment getter code
    logger.info(f"EA - Generating {EA_MAX_SP_EGC} special environment getter code.")
    new_tch, new_sp_egc_s = ea.gen_multi_sp_egc(
        count=EA_MAX_SP_EGC,
        in_con_path=in_con_path,
        genner=genner,
        backup_genner=backup_genner,
        docker_client=docker_client,
        test_container_id=test_container_id,
        model_name=BACKEND,
        max_attempts=EA_MAX_ATTEMPTS,
    )

    for t_message in new_tch.messages:
        if t_message.message.role == "assistant":
            logger.info(
                f"EA - Generated special environment getter code:  \n{t_message.message.content}"
            )

    logger.debug(f"EA - New TCH -\n{format_tch(new_tch)}")
    ea.tagged_chat_history.messages.extend(new_tch.messages)
    ea.sp_env_info_getter_codes.extend(new_sp_egc_s)

    # Execute all environment getting codes
    logger.info(f"EA - Executing all SP EGC code on container {main_container_id} ...")
    initial_special_env_infos = ea.execute_sp_egc_s(
        docker_client=docker_client, container_id=main_container_id
    )

    for special_env_info in initial_special_env_infos:
        logger.info(
            f"EA - Special environment info code execution result - \n{special_env_info}"
        )

    # Update environment agent's state
    logger.info("EA - Updating environment agent's SP EIH state...")
    ea.sp_env_info_history.append(initial_special_env_infos)

    # Initialize common agent
    main_ca = StrategyAgent(
        init_bs_env_info_history=ea.bs_env_info_history,
        init_sp_env_info_history=ea.sp_env_info_history,
        prev_strats=prev_strats,
        in_con_path=in_con_path,
    )

    new_tch = main_ca.get_initial_tch()

    logger.debug(f"SA - New TCH -\n{format_tch(new_tch)}")
    main_ca.tagged_chat_history.messages.extend(new_tch.messages)

    # Generate strategies based on the new environment info
    new_tch, main_ca.strats, strats_str = main_ca.gen_strats_(genner, BACKEND)
    logger.debug(f"SA - New TCH -\n{format_tch(new_tch)}")

    for t_message in new_tch.messages:
        if t_message.message.role == "assistant":
            logger.info(f"EA - Generated strat:  \n{t_message.message.content}")

    main_ca.tagged_chat_history.messages.extend(new_tch.messages)

    total_space_freed = 0
    total_success = 0
    list_of_ca: List[StrategyAgent] = []
    # Process each strategy
    for i, strat in enumerate(main_ca.strats):
        space_freed, strat_local_ca = process_strategy(
            strat=strat,
            i=i,
            ca=main_ca,
            ea=ea,
            docker_client=docker_client,
            genner=genner,
            main_container_id=main_container_id,
            test_container_id=test_container_id,
        )

        total_space_freed += space_freed
        total_success += 1 if space_freed > 0 else 0

        strat_local_ca.chosen_strat = strat
        strat_local_ca.space_freed = space_freed

        list_of_ca.append(strat_local_ca)

    timedate = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = save_folder / f"full_agent_data_{timedate}.json"

    main_ca.tagged_chat_history.messages.extend(new_tch.messages)

    save_list_of_agent_data(
        main_strat_agent=main_ca,
        copy_strat_agents=list_of_ca,
        env_agent=ea,
        space_freed=total_space_freed,
        save_path=save_path,
    )

    logger.info(
        f"{total_success} successful strats across {len(main_ca.strats)} strats, {total_space_freed} space freed across {len(main_ca.strats)} strats"
    )


def process_strategy(
    strat: str,
    i: int,
    ca: StrategyAgent,
    ea: EnvAgent,
    docker_client: docker.DockerClient,
    genner: Genner,
    main_container_id: str,
    test_container_id: str,
) -> Tuple[float, StrategyAgent]:
    """
    Process a single strategy, generating and testing code.

    Args:
        strat (str): The strategy to process.
        i (int): The index of the strategy.
        ca (StrategyAgent): The common agent instance.
        ea (EnvAgent): The environment agent instance.
        docker_client (docker.DockerClient): The Docker client.
        genner (callable): The code generator function.
        main_container_id (str): ID of the main Docker container.
        test_container_id (str): ID of the test Docker container.
        save_folder (Path): The folder to save data.
    """
    logger.info(f"SA - {i}-th strat - Code loop started on strat {strat}...")
    strat_local_ca = deepcopy(ca)
    # Get fresh environment info
    loop_bs_env_info = safe_detect_env(docker_client, main_container_id)

    # Update environment agent
    ea.bs_env_info_history.append(loop_bs_env_info)
    ea.sp_env_info_history.append(
        ea.execute_sp_egc_s(docker_client=docker_client, container_id=main_container_id)
    )

    # Update the environment info state of the common agent
    changes = strat_local_ca.update_env_info_state(
        ea.bs_env_info_history, ea.sp_env_info_history
    )
    logger.info(
        f"SA - {i}-th strat - Number of updates on common agent env info's state {changes}"
    )

    strat_local_ca.tagged_chat_history.messages.extend(
        get_strat_code_req_plist(strat=strat)
    )

    logger.info(strat_local_ca.tagged_chat_history.messages)

    # Generate and test code
    attempt = 0
    space_freed = 0
    while attempt < CA_MAX_ATTEMPTS:
        logger.debug(
            f"SA - {i}-th code - {attempt}-th attempt - "
            f"StrategyAgent's in loop chat history tag - \n{strat_local_ca.tagged_chat_history.messages}"
        )

        list_of_problems, code, raw_response = genner.generate_code(
            strat_local_ca.chat_history
        )

        logger.info(f"SA - Generated code: \n{raw_response}")

        if len(list_of_problems) > 0:
            logger.error(
                f"SA - {i}-th strat - {attempt}-th attempt - Failed to generate code. Local strat agent chat history tags: {strat_local_ca.tagged_chat_history.get_tags()}"
            )
            attempt += 1
            continue

        if not validate_and_run_code(
            code, docker_client, test_container_id, strat_local_ca, strat, i, attempt
        ):
            attempt += 1
            continue

        # The code is compile-able and can be executed in test container
        # Time to test it on real container
        container = docker_client.containers.get(main_container_id)
        exit_code, execution_output, reflected_code = run_code_in_con(
            container, code, "ea"
        )

        code_diffs: List[str] = get_code_diff(code, reflected_code)

        if len(code_diffs) > 0:
            logger.warning("SA - Generated code and in container are different.")
            for diff in code_diffs:
                logger.warning(f"SA - {diff}")
        else:
            logger.info("SA - Generated code and in container are the same.")

        fresh_basic_env_info = safe_detect_env(docker_client, main_container_id)
        space_freed, files_deleted = loop_bs_env_info.get_total_files_deleted(
            fresh_basic_env_info
        )

        if not files_deleted:
            logger.error(
                f"SA - {i}-th strat - {attempt + 1}-th attempt - "
                f"No spaces are freed \n{execution_output}"
            )
            logger.debug(f"SA - Code is \n{code[:100]}...")
            logger.debug(f"SA - In Container Code is \n{reflected_code[:100]}...")

            strat_local_ca.tagged_chat_history.messages.append(
                TaggedMessage(
                    message=Message(
                        role="assistant", content=f"```python\n{code}\n```"
                    ),
                    tag="genned_strat_code(deletion_fail)",
                )
            )
            strat_local_ca.tagged_chat_history.messages.extend(
                get_code_regen_plist(
                    task_description=f"Generate code to perform {strat}",
                    error_context="No files are being freed or deleted",
                    run_context="a Docker container",
                )
            )

            attempt += 1
            continue

        break

    if attempt >= CA_MAX_ATTEMPTS:
        logger.info(f"SA - {i}-th strat - Codegen loop ended up in failure")
    else:
        logger.info(f"SA - {i}-th strat - Codegen loop ended up in success")
        strat_local_ca.tagged_chat_history.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_strat_code(success)",
            )
        )

    logger.debug(f"SA - {i}-th strat - Code is \n{code[:100]}...")

    logger.info(
        f"Code loop is done for the strat {strat} on {attempt}-th iteration. Continuing to next strat if exists..."
    )

    return space_freed, strat_local_ca


def validate_and_run_code(
    code: str,
    docker_client: docker.DockerClient,
    test_container_id: str,
    copy_ca: StrategyAgent,
    strat: str,
    i: int,
    attempt: int,
) -> bool:
    """
    Validate and run the generated code in the test container.

    Args:
        code (str): The generated code to validate and run.
        docker_client (docker.DockerClient): The Docker client.
        test_container_id (str): ID of the test Docker container.
        copy_ca (StrategyAgent): The copy of the common agent.
        strat (str): The current strategy being processed.
        i (int): The index of the current strategy.
        attempt (int): The current attempt number.

    Returns:
        bool: True if the code is valid and runs successfully, False otherwise.
    """
    container = docker_client.containers.get(test_container_id)
    ast_valid, ast_error = is_valid_code_ast(code)

    if not ast_valid:
        logger.error(
            f"SA - {i}-th strat - {attempt + 1}-th attempt - "
            f"AST error \n{ast_error}"
        )
        logger.debug(f"SA - Code is \n{code[:100]}...")

        copy_ca.tagged_chat_history.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_strat_code(ast_fail)",
            )
        )
        copy_ca.tagged_chat_history.messages.extend(
            get_code_regen_plist(
                task_description=f"Generate code to perform {strat}",
                error_context=ast_error,
                run_context="a Python AST compiler",
            )
        )
        return False

    compile_valid, compiler_error = is_valid_code_compiler(code)
    if not compile_valid:
        logger.error(
            f"SA - {i}-th strat - {attempt + 1}-th attempt - "
            f"Native `compile` error \n{compiler_error}"
        )
        logger.debug(f"SA - Code is \n{code[:100]}...")

        copy_ca.tagged_chat_history.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_strat_code(compile_fail)",
            )
        )
        copy_ca.tagged_chat_history.messages.extend(
            get_code_regen_plist(
                task_description=f"Generate code to perform {strat}",
                error_context=compiler_error,
                run_context="a Python native compiler",
            )
        )
        return False

    exit_code, execution_output, reflected_code = run_code_in_con(container, code, "ea")

    if exit_code != 0:
        logger.error(
            f"SA - {i}-th strat - {attempt + 1}-th attempt - "
            f"In container error \n{execution_output}"
        )
        logger.debug(f"SA - Code is \n{code[:100]}...")
        logger.debug(f"SA - In Container Code is \n{reflected_code[:100]}...")

        copy_ca.tagged_chat_history.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_strat_code(container_fail)",
            )
        )
        copy_ca.tagged_chat_history.messages.extend(
            get_code_regen_plist(
                task_description=f"Generate code to perform {strat}",
                error_context=execution_output,
                run_context="a Docker container",
            )
        )
        return False

    if execution_output.strip() == "":
        logger.error(
            f"SA - {i}-th strat - {attempt + 1}-th attempt - "
            f"The code doesnt return any output"
        )
        logger.debug(f"SA - Code is \n{code[:100]}...")

        copy_ca.tagged_chat_history.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_strat_code(container_fail)",
            )
        )
        copy_ca.tagged_chat_history.messages.extend(
            get_code_regen_plist(
                task_description=f"Generate code to perform {strat}",
                error_context="No code output",
                run_context="a Docker container",
            )
        )
        return False

    return True


def wait_for_container(
    container_name: str, timeout: int = 30
) -> docker.models.containers.Container:
    """Wait for container to be fully running and return the container object."""
    client = docker.from_env()
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            container = client.containers.get(container_name)
            container.reload()

            if container.status == "running":
                # Test if container is truly ready by running a simple command
                exit_code, output = container.exec_run("ps aux")
                if exit_code == 0:
                    logger.info(f"Container {container_name} is fully running")
                    return container

            logger.debug(f"Container status: {container.status}, waiting...")
            time.sleep(1)

        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found yet, waiting...")
            time.sleep(1)
            continue

    raise TimeoutError(
        f"Container {container_name} did not start properly within {timeout} seconds"
    )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-as", "--always-success", action="store_true")
    args = parser.parse_args()

    try:
        process = subprocess.Popen(
            "docker compose up -d --build",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            shell=True,
            cwd="./docker/special-learn-compose",
        )

        assert process.stdout is not None
        for line in process.stdout:
            logger.info(f"{line.strip()}", end="")

        assert process.stderr is not None
        for line in process.stderr:
            logger.error(f"{line.strip()}", end="", file=sys.stderr)

        return_code = process.wait()

        if return_code != 0:
            print(f"Docker compose up launching process exited with code {return_code}")
            sys.exit(return_code)

        main_container_id = "special-learn-compose-service-a-1"
        test_container_id = "special-learn-compose-service-a-1"

        wait_for_container(main_container_id)

        main(
            debug=args.debug,
            always_success=args.always_success,
            main_container_id=main_container_id,
            test_container_id=test_container_id,
        )
    except Exception as e:
        raise e
