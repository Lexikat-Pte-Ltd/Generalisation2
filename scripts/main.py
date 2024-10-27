from argparse import ArgumentParser
from copy import deepcopy
import os
from pathlib import Path
import sys
from typing import List, Tuple

import docker
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
from loguru._defaults import LOGURU_FORMAT

from src.code import is_valid_code_ast, is_valid_code_compiler
from src.container import (
  run_code_in_con,
  safe_detect_env,
)
from src.data import EnvironmentInfo
from src.helper import (
  format_tch,
  format_tch_tags,
  get_code_diff,
  scan_json_files_for_strat,
  timed_input,
)
from src.prep import (
  get_code_regen_plist,
  get_strat_code_req_plist,
)
from src.gen import Genner, get_genner
from src.agent import EnvAgent, CommonAgent

# Load environment variables
load_dotenv()

# Constants (keeping original names)
EA_MAX_ATTEMPTS = 5
EA_MAX_SP_EGC = 2
CA_MAX_ATTEMPTS = 3

FEDORA_CONTAINER_ID = "fedora-learn-compose"
TEST_CONTAINER_ID = "mini-learn-compose"
BACKEND = "wizardcoder"
OAI_API_KEY = os.getenv("OPENAI_API_KEY")


def setup_logger(debug: bool) -> None:
  """
  Set up the logger with appropriate configuration.

  Args:
      debug (bool): Whether to enable debug mode.
  """
  logger.remove()
  logger.add(
    sys.stderr,
    backtrace=debug,
    diagnose=debug,
    format=LOGURU_FORMAT,  # type: ignore
    level="DEBUG" if debug else "INFO",
  )


def log_environment_variables() -> None:
  """Log the environment variables used in the script."""
  logger.info("Environment Variables:")
  logger.info(f"EA_MAX_ATTEMPTS: {EA_MAX_ATTEMPTS}")
  logger.info(f"EA_MAX_SP_EGC: {EA_MAX_SP_EGC}")
  logger.info(f"CA_MAX_ATTEMPTS: {CA_MAX_ATTEMPTS}")
  logger.info(f"FEDORA_CONTAINER_ID: {FEDORA_CONTAINER_ID}")
  logger.info(f"TEST_CONTAINER_ID: {TEST_CONTAINER_ID}")
  logger.info(f"BACKEND: {BACKEND}")
  logger.info(f"OAI_API_KEY: {'Set' if OAI_API_KEY else 'Not set'}")


def initialize_clients() -> Tuple[OpenAI, docker.DockerClient, Genner]:
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

  logger.info("Clients initialized:")
  logger.info(f"oai_client: {oai_client}")
  logger.info(f"docker_client: {docker_client}")
  logger.info(f"genner: {genner}")

  return oai_client, docker_client, genner


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
  oai_client, docker_client, genner = initialize_clients()

  save_folder = Path() / "data"
  prev_strats = scan_json_files_for_strat(save_folder)
  in_con_path = Path("/")

  logger.info(f"save_folder: {save_folder}")
  logger.info(f"prev_strats: {prev_strats}")
  logger.info(f"in_con_path: {in_con_path}")

  timed_input("Enter to continue or wait 3 seconds...")

  # Initialize environment agent
  logger.info("Initializing basic environment info.")
  initial_basic_env_info: EnvironmentInfo = safe_detect_env(
    docker_client, main_container_id
  )

  logger.info("EA - Initializing environment agent.")
  ea = EnvAgent(initial_basic_env_info=initial_basic_env_info, in_con_path=in_con_path)

  new_tch = ea.get_initial_tch()
  logger.debug(f"EA - New TCH -\n{format_tch(new_tch)}")
  ea.tagged_chat_history.extend(new_tch)

  # Generate special environment getter code
  logger.info(f"EA - Generating {EA_MAX_SP_EGC} special environment getter code.")
  new_tch, new_sp_egc_s = ea.gen_multi_sp_egc(
    count=EA_MAX_SP_EGC,
    in_con_path=in_con_path,
    genner=genner,
    docker_client=docker_client,
    test_container_id=test_container_id,
    max_attempts=EA_MAX_ATTEMPTS,
  )
  logger.debug(f"EA - New TCH -\n{format_tch(new_tch)}")
  ea.tagged_chat_history.extend(new_tch)
  ea.sp_env_info_getter_codes.extend(new_sp_egc_s)

  # Execute all environment getting codes
  logger.info(f"EA - Executing all SP EGC code on container {main_container_id} ...")
  initial_special_env_infos = ea.execute_sp_egc_s(
    docker_client=docker_client, container_id=main_container_id
  )

  # Update environment agent's state
  logger.info("EA - Updating environment agent's SP EIH state...")
  ea.sp_env_info_history.append(initial_special_env_infos)

  # Initialize common agent
  ca = CommonAgent(
    init_bs_env_info_history=ea.bs_env_info_history,
    init_sp_env_info_history=ea.sp_env_info_history,
    prev_strats=prev_strats,
    in_con_path=in_con_path,
  )

  new_tch = ca.get_initial_tch()
  logger.debug(f"EA - New TCH -\n{format_tch(new_tch)}")
  ca.tagged_chat_history.extend(new_tch)

  # Generate strategies based on the new environment info
  strats, raw_strats = ca.gen_strats(genner)

  if debug:
    strats = ["Clean up temporary files with commands like `rm -rf /tmp/*`."]
    raw_strats = str(strats)

  ca.tagged_chat_history.append(
    (
      {"role": "assistant", "content": f"```python\nlist = {raw_strats}\n```"},
      "gen_strats",
    )
  )

  # Process each strategy
  for i, strat in enumerate(strats):
    process_strategy(
      strat=strat,
      i=i,
      ca=ca,
      ea=ea,
      docker_client=docker_client,
      genner=genner,
      main_container_id=main_container_id,
      test_container_id=test_container_id,
      save_folder=save_folder,
    )


def process_strategy(
  strat: str,
  i: int,
  ca: CommonAgent,
  ea: EnvAgent,
  docker_client: docker.DockerClient,
  genner: Genner,
  main_container_id: str,
  test_container_id: str,
  save_folder: Path,
) -> None:
  """
  Process a single strategy, generating and testing code.

  Args:
      strat (str): The strategy to process.
      i (int): The index of the strategy.
      ca (CommonAgent): The common agent instance.
      ea (EnvAgent): The environment agent instance.
      docker_client (docker.DockerClient): The Docker client.
      genner (callable): The code generator function.
      main_container_id (str): ID of the main Docker container.
      test_container_id (str): ID of the test Docker container.
      save_folder (Path): The folder to save data.
  """
  logger.info(f"CA - {i}-th strat - Code loop started on strat {strat}...")
  copy_ca = deepcopy(ca)

  # Get fresh environment info
  loop_bs_env_info = safe_detect_env(docker_client, main_container_id)

  # Update environment agent
  ea.bs_env_info_history.append(loop_bs_env_info)
  ea.sp_env_info_history.append(
    ea.execute_sp_egc_s(docker_client=docker_client, container_id=main_container_id)
  )

  # Update the environment info state of the common agent
  changes = copy_ca.update_env_info_state(
    ea.bs_env_info_history, ea.sp_env_info_history
  )
  logger.info(
    f"CA - {i}-th strat - Number of updates on common agent env info's state {changes}"
  )

  # Save data before proceeding
  copy_ca.save_data(
    space_freed=-1,  # No space freed but can be useful maybe
    strat=strat,
    folder=save_folder,
  )
  logger.info(
    f"CA - {i}-th strat - Saved data @ {len(copy_ca.tagged_chat_history)} on {save_folder}."
  )

  # Prepare the common agent chat history with prep prompts
  copy_ca.tagged_chat_history.extend(get_strat_code_req_plist(strat=strat))

  # Generate and test code
  attempt = 0
  space_freed = 0
  while attempt < CA_MAX_ATTEMPTS:
    logger.debug(
      f"CA - {i}-th code - {attempt}-th attempt - "
      f"CommonAgent's in loop chat history tag - \n{format_tch_tags(copy_ca.tagged_chat_history)}"
    )

    code, raw_response = genner.generate_code(copy_ca.chat_history)

    if not validate_and_run_code(
      code, docker_client, test_container_id, copy_ca, strat, i, attempt
    ):
      attempt += 1
      continue

    # The code is compile-able and can be executed in test container
    # Time to test it on real container
    exit_code, execution_output, reflected_code = run_code_in_con(
      docker_client, main_container_id, code
    )

    code_diffs: List[str] = get_code_diff(code, reflected_code)

    if len(code_diffs) > 0:
      logger.warning("CA - Generated code and in container are different.")
      for diff in code_diffs:
        logger.warning(f"CA - {diff}")
    else:
      logger.info("CA - Generated code and in container are the same.")

    fresh_basic_env_info = safe_detect_env(docker_client, main_container_id)
    space_freed, files_deleted = loop_bs_env_info.total_files_deleted(
      fresh_basic_env_info
    )

    if not files_deleted:
      logger.error(
        f"CA - {i}-th strat - {attempt + 1}-th attempt - "
        f"No spaces are freed \n{execution_output}"
      )
      logger.debug(f"CA - Code is \n{code[:100]}...")
      logger.debug(f"CA - In Container Code is \n{reflected_code[:100]}...")

      copy_ca.tagged_chat_history.append(
        (
          {"role": "assistant", "content": f"```python\n{code}\n```"},
          "strat_codegen(deletion_fail)",
        )
      )
      copy_ca.tagged_chat_history.extend(
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
    logger.info(f"CA - {i}-th strat - Codegen loop ended up in failure")
  else:
    logger.info(f"CA - {i}-th strat - Codegen loop ended up in success")
    copy_ca.tagged_chat_history.append(
      (
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        "strat_codegen(success)",
      )
    )

  logger.debug(f"CA - {i}-th strat - Code is \n{code[:100]}...")

  copy_ca.save_data(
    space_freed=space_freed,
    strat=strat,
    folder=save_folder,
  )
  ea.save_data(folder=save_folder)

  logger.info(f"Code loop is done for the strat {strat} on {attempt}-th iteration.")
  logger.info("Continuing to next strat if exists...")


def validate_and_run_code(
  code: str,
  docker_client: docker.DockerClient,
  test_container_id: str,
  copy_ca: CommonAgent,
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
      copy_ca (CommonAgent): The copy of the common agent.
      strat (str): The current strategy being processed.
      i (int): The index of the current strategy.
      attempt (int): The current attempt number.

  Returns:
      bool: True if the code is valid and runs successfully, False otherwise.
  """
  ast_valid, ast_error = is_valid_code_ast(code)
  if not ast_valid:
    logger.error(
      f"CA - {i}-th strat - {attempt + 1}-th attempt - " f"AST error \n{ast_error}"
    )
    logger.debug(f"CA - Code is \n{code[:100]}...")

    copy_ca.tagged_chat_history.append(
      (
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        "strat_codegen(ast_fail)",
      )
    )
    copy_ca.tagged_chat_history.extend(
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
      f"CA - {i}-th strat - {attempt + 1}-th attempt - "
      f"Native `compile` error \n{compiler_error}"
    )
    logger.debug(f"CA - Code is \n{code[:100]}...")

    copy_ca.tagged_chat_history.append(
      (
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        "strat_codegen(compile_fail)",
      )
    )
    copy_ca.tagged_chat_history.extend(
      get_code_regen_plist(
        task_description=f"Generate code to perform {strat}",
        error_context=compiler_error,
        run_context="a Python native compiler",
      )
    )
    return False

  exit_code, execution_output, reflected_code = run_code_in_con(
    docker_client, test_container_id, code
  )

  if exit_code != 0:
    logger.error(
      f"CA - {i}-th strat - {attempt + 1}-th attempt - "
      f"In container error \n{execution_output}"
    )
    logger.debug(f"CA - Code is \n{code[:100]}...")
    logger.debug(f"CA - In Container Code is \n{reflected_code[:100]}...")

    copy_ca.tagged_chat_history.append(
      (
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        "strat_codegen(container_fail)",
      )
    )
    copy_ca.tagged_chat_history.extend(
      get_code_regen_plist(
        task_description=f"Generate code to perform {strat}",
        error_context=execution_output,
        run_context="a Docker container",
      )
    )
    return False

  if execution_output.strip() == "":
    logger.error(
      f"CA - {i}-th strat - {attempt + 1}-th attempt - "
      f"The code doesnt return any output"
    )
    logger.debug(f"CA - Code is \n{code[:100]}...")

    copy_ca.tagged_chat_history.append(
      (
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        "strat_codegen(container_fail)",
      )
    )
    copy_ca.tagged_chat_history.extend(
      get_code_regen_plist(
        task_description=f"Generate code to perform {strat}",
        error_context="No code output",
        run_context="a Docker container",
      )
    )
    return False

  return True


if __name__ == "__main__":
  parser = ArgumentParser(add_help=False)
  parser.add_argument("-d", "--debug", action="store_true")
  parser.add_argument("-as", "--always-success", action="store_true")
  args = parser.parse_args()

  main_container_id = (
    "fedora-learn-compose"
    if "fedora" == os.getenv("CONTAINER", "fedora")
    else "debian-learn-compose-dev"
  )
  test_container_id = "mini-learn-compose"

  main(
    debug=args.debug,
    always_success=args.always_success,
    main_container_id=main_container_id,
    test_container_id=test_container_id,
  )
