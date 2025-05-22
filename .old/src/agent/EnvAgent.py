from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple

from docker import DockerClient
from loguru import logger
from result import Result, Ok, Err, UnwrapError

from src.code import (
    is_valid_code_ast,
    is_valid_code_compiler,
)
from src.container import write_and_run_code_in_con_v2
from src.data import EnvironmentInfo
from src.genner import Genner
from src.helper import (
    get_code_diff,
    int_to_ordinal,
)
from src.prep import (
    get_basic_env_plist,
    get_code_regen_plist,
    get_special_egc_req_plist,
    get_special_egc_req_plist_fg,
    get_system_plist,
)
from src.types import Message, TaggedMessage, TaggedPList
from .BaseAgent import BaseAgent


class EnvAgent(BaseAgent):
    """Environment Agent for handling environment-related operations"""

    def __init__(
        self,
        initial_basic_env_infos: List[EnvironmentInfo],
        in_con_path: Path | str,
        model_name: str,  # Added model_name
    ):
        """Initialize environment agent to assists with main normal agent with special environment infos.

        Args:
            initial_basic_env_infos (List[EnvironmentInfo]):
                Initial basic environment info containing information about storage size.
            in_con_path (Path | str): In container work path.
            model_name (str): The name of the model being used.
        """
        super().__init__()
        self.sp_env_info_getter_codes: List[str] = []
        self.bs_env_info_history: List[EnvironmentInfo] = initial_basic_env_infos
        self.sp_env_info_history: List[List[str]] = []
        self.in_con_path = in_con_path
        self.model_name = model_name  # Store model_name

    def get_initial_tch(self) -> TaggedPList:
        local_tch = TaggedPList()

        local_tch.messages.extend(get_system_plist(in_con_path=self.in_con_path))
        local_tch.messages.extend(get_basic_env_plist(bs_eih=self.bs_env_info_history))

        return local_tch

    def debug_log_sp_eih(
        self, context="Logging Env Info's Special Env Info History ..."
    ) -> None:
        if context is not None:
            logger.debug(context)
        logger.debug(
            f"\n{self.sp_env_info_history}"
        )  # Corrected to sp_env_info_history

    def debug_log_bs_eih(
        self, context: str = "Logging Env Info's Basic Env Info History ..."
    ) -> None:
        if context is not None:
            logger.debug(context)
        logger.debug(f"\n{self.bs_env_info_history}")

    def gen_multi_sp_egc(
        self,
        code_count: int,
        genner: Genner,
        backup_genner: Genner,
        docker_client: DockerClient,
        test_container_id: str,
        model_name: str,  # This model_name is for the genner
        max_attempts_per_code: int = 5,
    ) -> Tuple[TaggedPList, List[str]]:
        local_tch = TaggedPList()
        sp_egc: List[str] = []
        cur_attempt = 0
        while len(sp_egc) < code_count:
            if cur_attempt > max_attempts_per_code:
                raise Exception(
                    "EnvAgent.gen_multi_sp_egc: Failed way too often, stopping the program...\n"
                    f"`cur_attempt`: {cur_attempt}\n"
                    f"`max_attempts_per_code`: {max_attempts_per_code}\n"
                    f"`len(sp_egc)`: {len(sp_egc)}\n"
                )
            cur_attempt += 1
            match self.gen_single_sp_egc(
                genner=genner,
                docker_client=docker_client,
                testing_container_id=test_container_id,
                model_name_for_prompt=model_name,  # Pass model_name for prompt generation
            ):
                case Ok((code, new_tch)):
                    sp_egc.append(code)
                    local_tch += new_tch
                    cur_attempt = 0
                case Err((err_msg, new_tch)):
                    local_tch += new_tch
                    logger.error(
                        "Failed generating special environment getter code, retrying, \n"
                        f"`model_name`: {model_name}\n"
                        f"`test_container_id`: {test_container_id}\n"
                        f"`cur_attempt`: {cur_attempt}\n"
                        f"`max_attempts_per_code`: {max_attempts_per_code}\n"
                        f"`err_msg`: \n{err_msg}\n",
                    )
                    continue

        return local_tch, sp_egc

    def gen_multi_sp_egc_2(
        self,
        code_count: int,
        genner: Genner,
        backup_genner: Genner,
        docker_client: DockerClient,
        test_container_id: str,
        model_name: str,  # This model_name is for the genner
        max_attempts_per_code: int = 5,
    ) -> Tuple[TaggedPList, List[str]]:
        local_tch = TaggedPList()
        sp_egc: List[str] = []
        cur_attempt = 0
        while len(sp_egc) < code_count:
            if cur_attempt > max_attempts_per_code:
                raise Exception(
                    "EnvAgent.gen_multi_sp_egc_2: Failed way too often, stopping the program...\n"
                    f"`cur_attempt`: {cur_attempt}\n"
                    f"`max_attempts_per_code`: {max_attempts_per_code}\n"
                    f"`len(sp_egc)`: {len(sp_egc)}\n"
                )
            cur_attempt += 1
            match self.gen_single_sp_egc_forced_guidance(
                genner=genner,
                docker_client=docker_client,
                testing_container_id=test_container_id,
                model_name_for_prompt=model_name,  # Pass model_name for prompt generation
            ):
                case Ok((code, new_tch)):
                    sp_egc.append(code)
                    local_tch += new_tch
                    cur_attempt = 0
                case Err((err_msg, new_tch)):
                    local_tch += new_tch
                    logger.error(
                        "Failed generating special environment getter code, retrying, \n"
                        f"`model_name`: {model_name}\n"
                        f"`test_container_id`: {test_container_id}\n"
                        f"`cur_attempt`: {cur_attempt}\n"
                        f"`max_attempts_per_code`: {max_attempts_per_code}\n"
                        f"`err_msg`: \n{err_msg}\n",
                    )
                    continue
        return local_tch, sp_egc

    def gen_multi_sp_egc_3(
        self,
        genner: Genner,
        docker_client: DockerClient,
        test_container_id: str,
        model_name: str,  # This model_name is for the genner
        max_attempts: int = 5,
    ) -> Tuple[TaggedPList, List[str]]:
        local_tch = TaggedPList()
        sp_egc: List[str] = []
        codes = [
            dedent("""
                import subprocess
                command = ["ip", "addr", "show"]
                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        print(f"Output of '{command}':")
                        print(stdout)
                    else:
                        print(f"Error running 'ip addr show', return code: {process.returncode}")
                        print(f"Error message: {stderr}")
                except FileNotFoundError:
                    print(f"Error: The '{command[0]}' command was not found. Make sure it's in your system's PATH.")
            """),
            dedent("""
                import subprocess
                command = ["nmap", "-sn", "172.25.0.0/24"]
                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        print(f"Output of '{command}':")
                        print(stdout)
                    else:
                        print(f"Error running '{command}', return code: {process.returncode}")
                        print(f"Error message: {stderr}")
                except FileNotFoundError:
                    print(f"Error: The '{command[0]}' command was not found. Make sure it's in your system's PATH.")
            """),
            dedent("""
                import subprocess
                command = ["ssh", "-v", "172.25.0.2", "-y"]
                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        print(f"Output of '{command}':")
                        print(stdout)
                    else:
                        print(f"Error running '{command}', return code: {process.returncode}")
                        print(f"Error message: {stderr}")
                except FileNotFoundError:
                    print(f"Error: The '{command[0]}' command was not found. Make sure it's in your system's PATH.")
            """),
        ]
        for code_content in codes:  # Renamed variable to avoid conflict
            match self.gen_single_sp_egc_forced_output(
                "```\n" + code_content.strip() + "\n```",
                code_content.strip(),
                genner=genner,
                docker_client=docker_client,
                testing_container_id=test_container_id,
                model_name_for_prompt=model_name,  # Pass model_name for prompt generation
            ):
                case Ok((env_getter_code, new_tch)):
                    sp_egc.append(env_getter_code)
                    local_tch += new_tch
                case Err((err_msg, new_tch)):
                    local_tch += new_tch
                    logger.error(
                        "Failed generating special environment getter code in gen_multi_sp_egc_3, \n"
                        f"`model_name`: {model_name}\n"
                        f"`test_container_id`: {test_container_id}\n"
                        f"`err_msg`: {err_msg}\n",
                    )
                    continue
        return local_tch, sp_egc

    def gen_single_sp_egc(
        self,
        genner: Genner,
        docker_client: DockerClient,
        testing_container_id: str,
        model_name_for_prompt: str,
    ) -> Result[Tuple[str, TaggedPList], Tuple[str, TaggedPList]]:
        local_tch: TaggedPList = TaggedPList()
        local_tch.messages.extend(
            get_special_egc_req_plist(
                in_con_path=self.in_con_path,
                model_name=model_name_for_prompt,
            )
        )
        try:
            code, _ = genner.generate_code(
                self.tagged_chat_history.as_plist() + local_tch.as_plist(),
            ).unwrap()

            # Write code in local file
            cur_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(
                f"./data/{model_name_for_prompt}/{model_name_for_prompt}_{cur_datetime}_ea_temp_sp_egc_code.py",
                "w",
            ) as f:
                f.write(code)

            match self.validate_code(code, docker_client, testing_container_id):
                case Ok(new_tch_validation):  # Renamed to avoid conflict
                    local_tch += new_tch_validation
                    return Ok((code, local_tch))
                case Err((e, new_tch_validation)):  # Renamed to avoid conflict
                    local_tch += new_tch_validation
                    return Err(
                        (
                            "EnvAgent.gen_single_sp_egc: Failed to validate code, \n"
                            f"`e`: {e}\n",
                            local_tch,
                        )
                    )
        except UnwrapError as e:
            return Err(
                (
                    "EnvAgent.gen_single_sp_egc: Failed to generate code, \n"
                    f"`e`: {e}\n",
                    TaggedPList(),
                )
            )

    def gen_single_sp_egc_forced_output(
        self,
        fake_response: str,
        fake_code: str,
        genner: Genner,
        docker_client: DockerClient,
        testing_container_id: str,
        model_name_for_prompt: str,
    ) -> Result[Tuple[str, TaggedPList], Tuple[str, TaggedPList]]:
        local_tch: TaggedPList = TaggedPList()
        local_tch.messages.extend(
            get_special_egc_req_plist(
                in_con_path=self.in_con_path,
                model_name=model_name_for_prompt,
            )
        )
        try:
            code = fake_code  # Use the provided fake_code
            # fake_response is used to construct the TaggedMessage if needed, not directly here
            match self.validate_code(code, docker_client, testing_container_id):
                case Ok(new_tch_validation):
                    local_tch += new_tch_validation
                    # Add the fake response as if the model generated it
                    local_tch.messages.append(
                        TaggedMessage(
                            message=Message(role="assistant", content=fake_response),
                            tag="genned_sp_egc(success_forced_output)",
                        )
                    )
                    return Ok((code, local_tch))
                case Err((e, new_tch_validation)):
                    local_tch += new_tch_validation
                    return Err(
                        (
                            "EnvAgent.gen_single_sp_egc_forced_output: Failed to validate code, \n"
                            f"`code`: {code}\n"
                            f"`e`: {e}\n",
                            local_tch,
                        )
                    )
        except Exception as e:  # General exception for safety
            return Err(
                (
                    "EnvAgent.gen_single_sp_egc_forced_output: Unexpected error, \n"
                    f"`e`: {e}\n",
                    local_tch,  # local_tch might have some messages
                )
            )

    def gen_single_sp_egc_forced_guidance(
        self,
        genner: Genner,
        docker_client: DockerClient,
        testing_container_id: str,
        model_name_for_prompt: str,
    ) -> Result[Tuple[str, TaggedPList], Tuple[str, TaggedPList]]:
        local_tch: TaggedPList = TaggedPList()
        local_tch.messages.extend(
            get_special_egc_req_plist_fg(
                in_con_path=self.in_con_path,
                model_name=model_name_for_prompt,
            )
        )
        try:
            code, _ = genner.generate_code(
                self.tagged_chat_history.as_plist() + local_tch.as_plist(),
            ).unwrap()
            match self.validate_code(code, docker_client, testing_container_id):
                case Ok(new_tch_validation):
                    local_tch += new_tch_validation
                    return Ok((code, local_tch))
                case Err((e, new_tch_validation)):
                    local_tch += new_tch_validation
                    return Err(
                        (
                            "EnvAgent.gen_single_sp_egc_forced_guidance: Failed to validate code, \n"
                            f"`e`: {e}\n",
                            local_tch,
                        )
                    )
        except UnwrapError as e:
            return Err(
                (
                    "EnvAgent.gen_single_sp_egc_forced_guidance: Failed to generate code, \n"
                    f"`e`: {e}\n",
                    TaggedPList(),
                )
            )

    def validate_code(
        self,
        code: str,
        docker_client: DockerClient,
        testing_container_id: str,
    ) -> Result[TaggedPList, Tuple[str, TaggedPList]]:  # model_name param removed
        local_tch: TaggedPList = TaggedPList()
        container = docker_client.containers.get(testing_container_id)

        ast_valid, ast_error = is_valid_code_ast(code)
        if not ast_valid:
            local_tch.messages.append(
                TaggedMessage(
                    message=Message(
                        role="assistant", content=f"```python\n{code}\n```"
                    ),
                    tag="genned_sp_egc(ast_fail)",
                )
            )
            local_tch.messages.extend(
                get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context=ast_error,
                    run_context="a Python AST compiler",
                )
            )
            return Err(
                (
                    "EnvAgent.validate_code: AST is not valid,\n"
                    f"`ast_error`: {ast_error}\n"
                    f"`testing_container_id`: {testing_container_id}\n"
                    f"`code`: {code}\n",
                    local_tch,
                )
            )

        compile_valid, compiler_error = is_valid_code_compiler(code)
        if not compile_valid:
            local_tch.messages.append(
                TaggedMessage(
                    message=Message(
                        role="assistant", content=f"```python\n{code}\n```"
                    ),
                    tag="genned_sp_egc(compile_fail)",
                )
            )
            local_tch.messages.extend(
                get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context=compiler_error,
                    run_context="a Python native compiler",
                )
            )
            return Err(
                (
                    "EnvAgent.validate_code: Code does not compile,\n"
                    f"`compiler_error`: {compiler_error}\n"
                    f"`testing_container_id`: {testing_container_id}\n"
                    f"`code`: {code}\n",
                    local_tch,
                )
            )

        exit_code, execution_output, reflected_code = write_and_run_code_in_con_v2(
            container,
            code,
            "ea",
            self.model_name,  # Use self.model_name
        )

        code_diffs = get_code_diff(code, reflected_code)
        if (
            len(code_diffs) > 0 and exit_code == 1
        ):  # Assuming exit_code 1 for this specific diff error
            return Err(
                (
                    "EnvAgent.validate_code: Different host code and in-container code,\n"
                    f"`code_diffs`: {code_diffs}\n"
                    f"`testing_container_id`: {testing_container_id}\n"
                    f"`code`: {code}\n",
                    local_tch,  # local_tch might be empty here, but consistent
                )
            )

        if exit_code != 0:
            local_tch.messages.append(
                TaggedMessage(
                    message=Message(
                        role="assistant", content=f"```python\n{code}\n```"
                    ),
                    tag="genned_sp_egc(container_fail)",
                )
            )
            local_tch.messages.extend(
                get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context=execution_output,
                    run_context="a Docker container",
                )
            )
            return Err(
                (
                    "EnvAgent.validate_code: Bad exit code when running it on container,\n"
                    f"`exit_code`: {exit_code}\n"
                    f"`execution_output`: {execution_output}\n"
                    f"`testing_container_id`: {testing_container_id}\n"
                    f"`code`: {code}\n",
                    local_tch,
                )
            )

        if execution_output.strip() == "":
            local_tch.messages.append(
                TaggedMessage(
                    message=Message(
                        role="assistant", content=f"```python\n{code}\n```"
                    ),
                    tag="genned_sp_egc(container_fail_empty_output)",  # More specific tag
                )
            )
            local_tch.messages.extend(
                get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context="No output was produced by the script.",  # Adjusted error
                    run_context="a Docker container",
                )
            )
            return Err(
                (
                    "EnvAgent.validate_code: Empty stdout when running on container,\n"
                    f"`exit_code`: {exit_code}\n"
                    f"`execution_output`: {execution_output}\n"
                    f"`testing_container_id`: {testing_container_id}\n"
                    f"`code`: {code}\n",
                    local_tch,
                )
            )

        local_tch.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_sp_egc(success)",
            )
        )
        return Ok(local_tch)

    def execute_sp_egc_s(
        self, docker_client: DockerClient, container_id: str
    ) -> List[str]:
        results = []
        container = docker_client.containers.get(container_id)
        for code in self.sp_env_info_getter_codes:
            _, container_result, reflected_code = write_and_run_code_in_con_v2(
                container,
                code,
                "ea",
                self.model_name,  # Use self.model_name
            )
            logger.debug(f"EA - Code is \n{code[:100]}...")
            code_diffs = get_code_diff(code, reflected_code)  # Corrected typo
            if len(code_diffs) > 0:
                logger.info(code_diffs)
                logger.warning("EA - Generated code and in container are different.")
            else:
                logger.info("EA - Generated code and in container are the same.")
            results.append(container_result)
        return results

    def append_new_code(self, code: str):
        self.sp_env_info_getter_codes.append(code)

    def as_native(self) -> Dict[str, Any]:
        return {
            "tagged_chat_history": self.tagged_chat_history.as_native(),
            "special_env_info_getter_codes": self.sp_env_info_getter_codes,
            "basic_env_info_history": [
                env_info.as_native() for env_info in self.bs_env_info_history
            ],
            "special_env_info_history": self.sp_env_info_history,
            "model_name": self.model_name,
        }
