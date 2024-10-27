from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger

from docker import DockerClient
from src.code import (
  is_valid_code_ast,
  is_valid_code_compiler,
)
from src.container import run_code_in_con
from src.data import EnvironmentInfo
from src.gen import Genner
from src.helper import (
  format_tch,
  format_tch_tags,
  get_code_diff,
  process_old_tch,
  represent_multiline_str,
  to_normal_plist,
)
from src.prep import (
  get_basic_env_plist,
  get_code_regen_plist,
  get_special_egc_req_plist,
  get_special_env_plist,
  get_strat_req_plist,
  get_system_plist,
)
from src.types import Message, TaggedMessage


class EnvAgent:
  """Flows are like this :

  1. Call `self.__init__`:
      Initial chat history `S(in_con_path)>U(initial_basic_env_info)>U(in_con_path)`
  2. Call `self.gen_new_special_env_getters`:
      Generate `S(...)>U(...)` as `sp_env_info_getter_codes`
  3. Call `self.update_special_env_getters`:
      Execute all `sp_env_info_getter_codes` and store it to `cur_sp_env_infos`
  4. Caller:
      Get `self.cur_sp_env_infos` and store it to `BasicAgent`'s `cur_sp_env_infos`
  """

  def __init__(
    self,
    initial_basic_env_info: EnvironmentInfo,
    in_con_path: Path | str,
  ):
    """Initilize environment agent to assists with main normal agent with special environment infos.

    Args:
        initial_basic_env_info (EnvironmentInfo): Initial basic environment info containing
            information about storage size.
        in_con_path (Path | str): In container work path.
    """

    self.sp_env_info_getter_codes: List[str] = []
    self.tagged_chat_history: List[TaggedMessage] = []

    self.bs_env_info_history: List[EnvironmentInfo] = []
    self.bs_env_info_history.append(initial_basic_env_info)
    self.sp_env_info_history: List[List[str]] = []

    self.in_con_path = in_con_path

  def get_initial_tch(self) -> List[TaggedMessage]:
    local_tch = []

    local_tch.extend(  #
      get_system_plist(in_con_path=self.in_con_path)
    )
    local_tch.extend(  #
      get_basic_env_plist(bs_eih=self.bs_env_info_history)
    )

    return local_tch

  @property
  def chat_history(self) -> List[Message]:
    """Use this property function if you want to get the more ChatGPT-like chat history of this agent.

    Returns:
        List[Message]: OpenAI standard chat history.
    """
    return to_normal_plist(self.tagged_chat_history)

  def debug_log_tch(self, context="Logging Env Info's TCH ...") -> None:
    """Use this to log TCH (Tagged Chat History)

    Args:
        context (str, optional): Extra log before the main log.
    """

    if context is not None:
      logger.debug(context)

    logger.debug(format_tch(self.tagged_chat_history))

  def debug_log_sp_eih(
    self, context="Logging Env Info's Special Env Info History ..."
  ) -> None:
    """Use this to log SP EIH (Special Environment Info History)

    Args:
        context (str, optional): Extra log before the main log.
    """
    if context is not None:
      logger.debug(context)

    logger.debug(f"\n{self.bs_env_info_history}")

  def debug_log_bs_eih(
    self, context: str = "Logging Env Info's Basic Env Info History ..."
  ) -> None:
    """Use this to log BS EIH (Basic Environment Info History)

    Args:
        context (str, optional): Extra log before the main log.
    """

    if context is not None:
      logger.debug(context)

    logger.debug(f"\n{self.bs_env_info_history}")

  def gen_multi_sp_egc(
    self,
    count: int,
    in_con_path: str | Path,
    genner: Genner,
    docker_client: DockerClient,
    test_container_id: str,
    max_attempts: int = 5,
  ) -> Tuple[List[TaggedMessage], List[str]]:
    """Generate multiple SP EGC (Special Environment Info Getter Code)

    Args:
        count (int)
        in_con_path (str | Path)
        genner (Callable)
        docker_client (DockerClient)
        testing_container_id (str)
        max_attempts (int, optional): Defaults to 5.

    Returns:
        List[TaggedMessage]: TCH to append to EIH's TCH
        List[str]: New SP EGC to append to EIH's SP EGC.

    Raises:
        Exception: _description_
    """
    local_tch: List[TaggedMessage] = []
    sp_egc: List[str] = []

    for i in range(count):
      local_tch.extend(get_special_egc_req_plist(in_con_path))

      env_getter_code, succeed, new_tch = self.gen_single_sp_egc(
        count=i,
        primary_genner=genner,
        docker_client=docker_client,
        testing_container_id=test_container_id,
        max_attempts=max_attempts,
      )

      # If special environment getters generation process failed
      if not succeed:
        logger.error(
          f"EA - {i}-th SP EGC - Failed generating env getter code after {max_attempts}"
        )
        raise Exception("EA - Somehow `gen_sp_egc` failed.")

      # Update (append) env_agent's state
      logger.info(f"EA - {i}-th SP EGC - Appending env_agent with new env getter code.")

      local_tch.extend(new_tch)
      sp_egc.append(env_getter_code)

    return local_tch, sp_egc

  def gen_single_sp_egc(
    self,
    count: int,
    primary_genner: Genner,
    backup_genner: Genner,
    docker_client: DockerClient,
    testing_container_id: str,
    max_attempts: int = 5,
    max_primary_attempts: int = 3,  # New parameter
  ) -> Tuple[str, bool, List[TaggedMessage]]:
    """Generate and test SP EGC (Special Env Info Getter Code).

    Args:
        count (int): Current count for debugging purposes.
        genner (GennerFunction): Primary genner function.
        docker_client (DockerClient): Docker Client.
        testing_container_id (str): Container ID to evaluate generated code.
        max_attempts (int): Maximum number of total generation attempts.
        max_primary_attempts (int): Maximum number of attempts with primary genner before switching.

    Returns:
        Tuple[str, bool, List[TaggedMessage]]: Generated code, success status, and new tagged chat history.
    """
    local_tch: List[TaggedMessage] = []
    local_tch.extend(get_special_egc_req_plist(self.in_con_path))

    primary_attempts = 0
    for attempt in range(max_attempts):
      logger.debug(
        f"EA - {count}-th code - {attempt}-th attempt - "
        f"EnvAgent's in loop chat history tags - \n {format_tch_tags(self.tagged_chat_history)}"
      )

      if primary_attempts >= max_primary_attempts:
        logger.warning(
          f"Switching to backup genner after {primary_attempts} failed primary attempts."
        )
        current_genner = backup_genner
      else:
        current_genner = primary_genner

      try:
        code, raw_response = current_genner.generate_code(
          to_normal_plist(self.tagged_chat_history) + to_normal_plist(local_tch),
        )
      except Exception as e:
        logger.error(f"Code generation failed: {e}")
        primary_attempts += 1
        continue

      if not self.validate_code(
        code, docker_client, testing_container_id, local_tch, count, attempt
      ):
        primary_attempts += 1
        continue

      logger.info(f"EA - {count}-th code - Codegen loop succeed")
      logger.info(f"EA - {count}-th code - Cache TCH is {len(local_tch)}")
      logger.debug(f"EA - {count}-th code - Code is \n{code}")

      return code, True, local_tch

    return "", False, local_tch

  def validate_code(
    self, code, docker_client, testing_container_id, local_tch, count, attempt
  ):
    ast_valid, ast_error = is_valid_code_ast(code)
    if not ast_valid:
      logger.error(
        f"EA - {count}-th code - {attempt}-th attempt - AST error \n{ast_error}"
      )
      logger.debug(f"EA - Code is \n{code[:100]}...")
      local_tch.append(
        (
          {"role": "assistant", "content": f"```python\n{code}\n```"},
          "gen_sp_egc(ast_fail)",
        )
      )
      local_tch.extend(
        get_code_regen_plist(
          task_description="Generate and test code for getting non-basic environment info",
          error_context=ast_error,
          run_context="a Python AST compiler",
        )
      )
      return False

    compile_valid, compiler_error = is_valid_code_compiler(code)
    if not compile_valid:
      logger.error(
        f"EA - {count}-th code - {attempt}-th attempt - Native `compile` error \n{compiler_error}"
      )
      logger.debug(f"EA - Code is \n{code[:100]}...")
      local_tch.append(
        (
          {"role": "assistant", "content": f"```python\n{code}\n```"},
          "gen_sp_egc(compile_fail)",
        )
      )
      local_tch.extend(
        get_code_regen_plist(
          task_description="Generate and test code for getting non-basic environment info",
          error_context=compiler_error,
          run_context="a Python native compiler",
        )
      )
      return False

    exit_code, execution_output, reflected_code = run_code_in_con(
      docker_client, testing_container_id, code
    )

    code_diffs = get_code_diff(code, reflected_code)
    if len(code_diffs) > 0:
      logger.error(code_diffs)
      raise Exception("EA - Generated code and in container are different.")
    else:
      logger.info("EA - Generated code and in container are the same.")

    if exit_code != 0:
      logger.error(
        f"EA - {count}-th code - {attempt}-th attempt - In container error \n{execution_output}"
      )
      logger.debug(f"EA - Code is \n{code[:100]}...")
      local_tch.append(
        (
          {"role": "assistant", "content": f"```python\n{code}\n```"},
          "gen_sp_egc(container_fail)",
        )
      )
      local_tch.extend(
        get_code_regen_plist(
          task_description="Generate and test code for getting non-basic environment info",
          error_context=execution_output,
          run_context="a Docker container",
        )
      )
      return False

    if execution_output.strip() == "":
      logger.error(
        f"EA - {count}-th code - {attempt}-th attempt - The code doesnt return any output"
      )
      logger.debug(f"EA - Code is \n{code[:100]}...")
      local_tch.append(
        (
          {"role": "assistant", "content": f"```python\n{code}\n```"},
          "gen_sp_egc(container_fail)",
        )
      )
      local_tch.extend(
        get_code_regen_plist(
          task_description="Generate and test code for getting non-basic environment info",
          error_context="No files are being freed or deleted",
          run_context="a Docker container",
        )
      )
      return False

    local_tch.append(
      (
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        "gen_sp_egc(success)",
      )
    )
    return True

  def execute_sp_egc_s(
    self, docker_client: DockerClient, container_id: str
  ) -> List[str]:
    """Given a docker client and container ID, will execute stored `self.sp_env_info_getter_codes`
    on `container_id` and store the result on `self.cur_sp_env_info`.

    Args:
        docker_client (DockerClient): Docker Client.
        container_id (str): Container ID to execute codes (env_info_getter) on.

    Returns:
        results (List[str]): List of current special environment infos.
    """
    results = []

    for code in self.sp_env_info_getter_codes:
      _, container_result, reflected_code = run_code_in_con(
        docker_client, container_id, code
      )

      logger.debug(f"EA - Code is \n{code[:100]}...")

      code_diffs = get_code_diff(code, reflected_code)

      if len(code_diffs) > 0:
        logger.info(code_diffs)
        logger.warning("EA - Generated code and in container are different.")
      else:
        logger.info("EA - Generated code and in container are the same.")

      results.append(container_result)

    return results

  def append_new_code(self, code: str, tag="special_env_getter_code"):
    """Appends `({"role": "assistant", "content": code}, tag)` to this agent's chat history and
    appends new code to this agent's `sp_env_info_getter_codes`

    Args:
        code (str): Newly generated chat history.
        tag: Tag to be used in `TaggedPList`.
    """
    self.sp_env_info_getter_codes.append(code)

  def save_data(self, folder: Path | str):
    yaml.add_representer(str, represent_multiline_str)

    tch_len = len(self.tagged_chat_history)
    formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H:%M")
    file_name = f"ea_{formatted_datetime}_run_data_at_{tch_len}.yaml"

    save_data = {
      "tag": "[history, env_agent]",
      "sp_egc_s": self.sp_env_info_getter_codes,
      "tagged_chat_history": process_old_tch(self.tagged_chat_history),
    }

    with open(Path(folder) / file_name, "w") as yaml_file:
      yaml.dump(
        save_data,
        yaml_file,
        width=100,
        default_flow_style=False,
        sort_keys=False,
      )


class CommonAgent:
  """Flows are like this:
  1. `__init__`:
      Init chat history with `S(initial_basic_env_info, in_con_path)>U(cur_sp_env)>U(in_con_path)`
  2. `gen_strats`:
      Generate `S(...)>U(...)>U(...)` as `cur_strats`
  3. `update_special_env_getters`:
      Execute all `sp_env_info_getter_codes` and store it to `cur_sp_env_infos`
  """

  def __init__(
    self,
    init_bs_env_info_history: List[EnvironmentInfo],
    init_sp_env_info_history: List[List[str]],
    prev_strats: List[str],
    in_con_path: str | Path,
  ):
    """Initialize a common agent to be used for strategies generation and space clearing code generation.

    Args:
        initial_basic_env_info (EnvironmentInfo): Initial basic environment info, will only
            be stored in chat history.
        initial_special_env_infos (List[str]): Initial special environment info, generated from `EnvAgent`
            will only be stored in chat history.
        in_con_path (str | Path): In container working path.
    """

    self.tagged_chat_history: List[TaggedMessage] = []
    self.strats: List[str] = []

    self.in_con_path = in_con_path

    self.init_bs_env_info_history = init_bs_env_info_history
    self.init_sp_env_info_history = init_sp_env_info_history
    self.prev_strats = prev_strats

  def get_initial_tch(self) -> List[TaggedMessage]:
    local_tch = []

    # Add system prompt
    local_tch.extend(  #
      get_system_plist(in_con_path=self.in_con_path)
    )
    # Add basic env inclusion prompt
    local_tch.extend(  #
      get_basic_env_plist(bs_eih=self.init_bs_env_info_history)
    )
    # Add special env inclusion prompt
    local_tch.extend(  #
      get_special_env_plist(sp_eih=self.init_sp_env_info_history)
    )
    # Add strategy gen prompt
    local_tch.extend(  #
      get_strat_req_plist(in_con_path=self.in_con_path, prev_strats=self.prev_strats)
    )

    return local_tch

  @property
  def chat_history(self) -> List[Message]:
    """Use this property function if you want to get the more ChatGPT-like chat history of this agent.

    Returns:
        List[Message]: OpenAI standard chat history.
    """

    return to_normal_plist(self.tagged_chat_history)

  def debug_log(self) -> None:
    """Use this to print chat history"""

    logger.debug(format_tch(self.tagged_chat_history))

  def gen_strats(self, genner: Genner) -> Tuple[List[str], str]:
    """Given an OAI client, will generate strategies based on the current chat history
    containing system prompt, initial environment info, and special environment info found by
    executing special environment info getters on a specified container.

    Args:
        oai_client: OpenAI client.

    Returns:
        strategies: List of string that are strategies.
    """

    return genner.generate_list(self.tagged_chat_history)

  def update_env_info_state(
    self,
    fresh_bs_eih: List[EnvironmentInfo],
    fresh_sp_eih: List[List[str]],
    basic_env_info_tag="get_basic_env_plist",
    special_env_info_tag="get_special_env_plist",
  ) -> int:
    changes = 0

    for i in range(len(self.tagged_chat_history)):
      tag = self.tagged_chat_history[i][1]

      if tag == basic_env_info_tag:
        self.tagged_chat_history[i] = get_basic_env_plist(fresh_bs_eih)[0]

        changes += 1

        continue
      elif tag == special_env_info_tag:
        self.tagged_chat_history[i] = get_special_env_plist(fresh_sp_eih)[0]

        changes += 1

        continue

    return changes

  def save_data(self, space_freed: float, strat: str, folder: Path | str):
    import yaml

    yaml.add_representer(str, represent_multiline_str)

    strat_identifier = "".join(
      [word[0].lower() for word in strat.split(" ") if word.isalpha()][:10]
    )
    tch_len = len(self.tagged_chat_history)
    formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H:%M")

    file_name = f"ca_{formatted_datetime}_run_data_at_{tch_len}_{strat_identifier}.yaml"

    # Save the sequential one
    save_data = {
      "tag": "[history, common_agent]",
      "strat": strat,
      "space_freed": space_freed,
      "tagged_chat_history": process_old_tch(self.tagged_chat_history),
    }
    # save_data = OrderedDict(
    #     [
    #         ("tag", "[history, common_agent]"),
    #         ("strat", strat),
    #         ("space_freed", space_freed),
    #         ("tagged_chat_history", self.tagged_chat_history),
    #     ]
    # )

    with open(Path(folder) / file_name, "w") as yaml_file:
      yaml.dump(
        save_data,
        yaml_file,
        width=100,
        default_flow_style=False,
        sort_keys=False,
      )
