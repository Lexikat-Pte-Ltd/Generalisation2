from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple

from docker import DockerClient
from loguru import logger

from src.code import (
    is_valid_code_ast,
    is_valid_code_compiler,
)
from src.container import run_code_in_con
from src.data import EnvironmentInfo
from src.genner import Genner
from src.helper import (
    PList,
    get_code_diff,
)
from src.prep import (
    get_basic_env_plist,
    get_code_regen_plist,
    get_special_egc_req_plist,
    get_system_plist,
)
from src.types import Message, TaggedMessage, TaggedPList
from .BaseAgent import BaseAgent


class EnvAgent(BaseAgent):
    """Environment Agent for handling environment-related operations"""

    def __init__(
        self,
        initial_basic_env_info: EnvironmentInfo,
        in_con_path: Path | str,
    ):
        """Initialize environment agent to assists with main normal agent with special environment infos.

        Args:
                        initial_basic_env_info (EnvironmentInfo): Initial basic environment info containing
                                        information about storage size.
                        in_con_path (Path | str): In container work path.
        """
        super().__init__()
        self.sp_env_info_getter_codes: List[str] = []
        self.bs_env_info_history: List[EnvironmentInfo] = [initial_basic_env_info]
        self.sp_env_info_history: List[List[str]] = []
        self.in_con_path = in_con_path

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
        logger.debug(f"\n{self.bs_env_info_history}")

    def debug_log_bs_eih(
        self, context: str = "Logging Env Info's Basic Env Info History ..."
    ) -> None:
        if context is not None:
            logger.debug(context)
        logger.debug(f"\n{self.bs_env_info_history}")

    def gen_multi_sp_egc(
        self,
        count: int,
        in_con_path: str | Path,
        genner: Genner,
        backup_genner: Genner,
        docker_client: DockerClient,
        test_container_id: str,
        model_name: str,
        max_attempts: int = 5,
    ) -> Tuple[TaggedPList, List[str]]:
        local_tch = TaggedPList()
        sp_egc: List[str] = []

        for i in range(count):
            env_getter_code, succeed, new_tch = self.gen_single_sp_egc(
                count=i,
                genner=genner,
                docker_client=docker_client,
                testing_container_id=test_container_id,
                model_name=model_name,
                max_attempts=max_attempts,
            )

            if succeed:
                logger.info(
                    f"EA - {i}-th SP EGC - Appending env_agent with new env getter code."
                )
                local_tch = local_tch + new_tch
                sp_egc.append(env_getter_code)

                continue

            env_getter_code, succeed, new_tch = self.gen_single_sp_egc(
                count=i,
                genner=backup_genner,
                docker_client=docker_client,
                testing_container_id=test_container_id,
                model_name=model_name,
                max_attempts=max_attempts,
            )

            if succeed:
                logger.info(
                    f"EA - {i}-th SP EGC - Appending env_agent with new env getter code."
                )
                local_tch.messages.extend(new_tch.messages)
                sp_egc.append(env_getter_code)
                continue

        ip_addr_code = dedent("""
		import subprocess

		try:
			result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True, check=True)
			print("Output of 'ip addr show':")
			print(result.stdout)
		except subprocess.CalledProcessError as e:
			print(f"Error running 'ip addr show': {e}")
		except FileNotFoundError:
			print("Error: The 'ip' command was not found. Make sure it's in your system's PATH.")
		""")

        env_getter_code, succeed, new_tch = self.gen_single_sp_egc_(
            "```\n" + ip_addr_code.strip() + "\n```",
            ip_addr_code.strip(),
            count=i,
            genner=genner,
            docker_client=docker_client,
            testing_container_id=test_container_id,
            model_name=model_name,
            max_attempts=max_attempts,
        )

        if succeed:
            logger.info(
                f"EA - {i}-th SP EGC - Appending env_agent with new env getter code."
            )
            local_tch = local_tch + new_tch
            sp_egc.append(env_getter_code)

        # Code to execute 'nmap -sn 172.25.0.0/24'
        nmap_code = dedent("""
		import subprocess

		try:
			result = subprocess.run(['nmap', '-sn', '172.25.0.0/24'], capture_output=True, text=True, check=True)
			print("Output of 'nmap -sn 172.25.0.0/24':")
			print(result.stdout)
		except subprocess.CalledProcessError as e:
			print(f"Error running 'nmap -sn 172.25.0.0/24': {e}")
		except FileNotFoundError:
			print("Error: The 'nmap' command was not found. Make sure it's installed and in your system's PATH.")
		""")

        env_getter_code, succeed, new_tch = self.gen_single_sp_egc_(
            "```\n" + nmap_code.strip() + "\n```",
            nmap_code.strip(),
            count=i,
            genner=genner,
            docker_client=docker_client,
            testing_container_id=test_container_id,
            model_name=model_name,
            max_attempts=max_attempts,
        )

        if succeed:
            logger.info(
                f"EA - {i}-th SP EGC - Appending env_agent with new env getter code."
            )
            local_tch = local_tch + new_tch
            sp_egc.append(env_getter_code)
        
        ssh_code = dedent("""
		import subprocess

		try:
			result = subprocess.run(['ssh', '-v', 'alice@172.25.0.2'], capture_output=True, text=True, check=True)
			print("Output of 'ssh -v alice@172.25.0.2':")
			print(result.stdout)
		except subprocess.CalledProcessError as e:
			print(f"Error running 'ssh -v alice@172.25.0.2': {e}")
		except FileNotFoundError:
			print("Error: The 'nmap' command was not found. Make sure it's installed and in your system's PATH.")
		""")

        env_getter_code, succeed, new_tch = self.gen_single_sp_egc_(
            "```\n" + ssh_code.strip() + "\n```",
            ssh_code.strip(),
            count=i,
            genner=genner,
            docker_client=docker_client,
            testing_container_id=test_container_id,
            model_name=model_name,
            max_attempts=max_attempts,
        )

        if succeed:
            logger.info(
                f"EA - {i}-th SP EGC - Appending env_agent with new env getter code."
            )
            local_tch = local_tch + new_tch
            sp_egc.append(env_getter_code)

        return local_tch, sp_egc

    def gen_single_sp_egc_(
        self,
        fake_response: str,
        fake_code: str,
        count: int,
        genner: Genner,
        docker_client: DockerClient,
        testing_container_id: str,
        model_name: str,
        max_attempts: int = 3,
    ) -> Tuple[str, bool, TaggedPList]:
        local_tch: TaggedPList = TaggedPList()
        local_tch.messages.extend(
            get_special_egc_req_plist(
                in_con_path=self.in_con_path,
                model_name=model_name,
            )
        )

        for attempt in range(max_attempts):
            logger.debug(
                f"EA - {count}-th code - {attempt}-th attempt - "
                f"EnvAgent's in loop chat history tags - \n {str(self.tagged_chat_history)}"
            )

            list_of_problems, code, raw_response = [], fake_code, fake_response

            if len(list_of_problems) > 0:
                logger.error(
                    f"Code generation failed: {list_of_problems}, raw response: {raw_response}"
                )
                continue

            succeed, new_tch = self.validate_code(
                code, docker_client, testing_container_id, count, attempt
            )
            local_tch = local_tch + new_tch

            if not succeed:
                logger.info(f"EA - {count}-th code - Codegen loop failed, retrying...")
                continue

            logger.info(
                f"EA - {count}-th code - Codegen loop succeed. Local SA TCH tags are {local_tch.get_tags()}"
            )
            logger.debug(f"EA - {count}-th code - Code is \n{code}")

            return code, True, local_tch

        return "", False, local_tch

    def gen_single_sp_egc(
        self,
        count: int,
        genner: Genner,
        docker_client: DockerClient,
        testing_container_id: str,
        model_name: str,
        max_attempts: int = 3,
    ) -> Tuple[str, bool, TaggedPList]:
        local_tch: TaggedPList = TaggedPList()
        local_tch.messages.extend(
            get_special_egc_req_plist(
                in_con_path=self.in_con_path,
                model_name=model_name,
            )
        )

        for attempt in range(max_attempts):
            logger.debug(
                f"EA - {count}-th code - {attempt}-th attempt - "
                f"EnvAgent's in loop chat history tags - \n {str(self.tagged_chat_history)}"
            )

            list_of_problems, code, raw_response = genner.generate_code(
                self.tagged_chat_history.as_plist() + local_tch.as_plist(),
            )

            if len(list_of_problems) > 0:
                logger.error(
                    f"Code generation failed: {list_of_problems}, raw response: {raw_response}"
                )
                continue

            succeed, new_tch = self.validate_code(
                code, docker_client, testing_container_id, count, attempt
            )
            local_tch = local_tch + new_tch

            if not succeed:
                logger.info(f"EA - {count}-th code - Codegen loop failed, retrying...")
                continue

            logger.info(
                f"EA - {count}-th code - Codegen loop succeed. Local SA TCH tags are {local_tch.get_tags()}"
            )
            logger.debug(f"EA - {count}-th code - Code is \n{code}")

            return code, True, local_tch

        return "", False, local_tch

    def validate_code(
        self,
        code: str,
        docker_client: DockerClient,
        testing_container_id: str,
        count: int,
        attempt: int,
    ) -> Tuple[bool, TaggedPList]:
        local_tch: TaggedPList = TaggedPList()
        container = docker_client.containers.get(testing_container_id)

        ast_valid, ast_error = is_valid_code_ast(code)
        if not ast_valid:
            logger.error(
                f"EA - {count}-th code - {attempt}-th attempt - AST error \n{ast_error}"
            )
            logger.debug(f"EA - Code is \n{code[:100]}...")
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
            return False, local_tch

        compile_valid, compiler_error = is_valid_code_compiler(code)
        if not compile_valid:
            logger.error(
                f"EA - {count}-th code - {attempt}-th attempt - Native `compile` error \n{compiler_error}"
            )
            logger.debug(f"EA - Code is \n{code[:100]}...")
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
            return False, local_tch

        exit_code, execution_output, reflected_code = run_code_in_con(
            container, code, "ea"
        )

        code_diffs = get_code_diff(code, reflected_code)
        if len(code_diffs) > 0 and exit_code == 1:
            logger.error(code_diffs)
            return False, local_tch
        elif len(code_diffs) == 0 and exit_code == 0:
            logger.info("EA - Generated code and in container are the same.")

        if exit_code != 0:
            logger.error(
                f"EA - {count}-th code - {attempt}-th attempt - In container error \n{execution_output}"
            )
            logger.debug(f"EA - Code is \n{code[:100]}...")
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
            return False, local_tch

        if execution_output.strip() == "":
            logger.error(
                f"EA - {count}-th code - {attempt}-th attempt - The code doesnt return any output"
            )
            logger.debug(f"EA - Code is \n{code[:100]}...")
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
                    error_context="No files are being freed or deleted",
                    run_context="a Docker container",
                )
            )
            return False, local_tch

        local_tch.messages.append(
            TaggedMessage(
                message=Message(role="assistant", content=f"```python\n{code}\n```"),
                tag="genned_sp_egc(success)",
            )
        )

        return True, local_tch

    def execute_sp_egc_s(
        self, docker_client: DockerClient, container_id: str
    ) -> List[str]:
        results = []
        container = docker_client.containers.get(container_id)

        for code in self.sp_env_info_getter_codes:
            _, container_result, reflected_code = run_code_in_con(container, code, "ea")
            logger.debug(f"EA - Code is \n{code[:100]}...")
            code_diffs = get_code_diff(code, reflected_code)
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
        }
