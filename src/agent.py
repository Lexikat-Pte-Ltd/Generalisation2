import copy
import json
from pathlib import Path
from typing import Callable, List, Tuple

from loguru import logger
from docker import DockerClient

from src.code import (
    is_valid_code_ast,
    is_valid_code_compiler,
)
from src.helper import format_tch, to_normal_plist
from src.data import EnvironmentInfo
from src.prep import (
    BASIC_ENV_PLIST_TAG,
    SPECIAL_EGC_REQ_PLIST_TAG,
    SPECIAL_ENV_PLIST_TAG,
    SYSTEM_PLIST_TAG,
    get_basic_env_plist,
    get_code_regen_plist,
    get_strat_req_plist,
    get_system_plist,
    get_special_env_plist,
)
from src.gen import gen_code, gen_list
from src.types import Message, TaggedMessage
from src.container import run_code_in_con


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

        self.bs_env_info_history = [initial_basic_env_info]
        self.sp_env_info_history: List[str] = []

        self.in_con_path = in_con_path

        self.tagged_chat_history += get_system_plist(in_con_path=str(in_con_path))
        self.tagged_chat_history += get_basic_env_plist(
            basic_env_info=str(initial_basic_env_info)
        )

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

    def gen_sp_egc(
        self,
        genner: Callable,
        docker_client: DockerClient,
        testing_container_id: str,
        max_attempts: int = 5,
    ) -> Tuple[str, bool]:
        """Generate and test SP EGC (Special Env Info Getter Code).

        ```
        Possible Flows :
        1.
            Start State :
                Special env not yet acquired.
            Flow :
                System > Basic Env > Gen Request > Failed Gen > Regen Request > Failed / Success Gen
            End State :
                SP EGC and special env acquired.
        2.
            Start State :
                Anything that leads to a failed gen.
            Flow :
                - ... > Failed Gen > Regen Request > Failed / Success Gen
            Start State :
                - Failed / success gen.
        ```

        Args:
            genner (GennerFunction): Genner function.
            docker_client (DockerClient): Docker Client.
            testing_container_id (str): Container ID to evaluate generated code.
            max_attempts (int): Maximum number of generation attempts.

        Returns:
            Tuple[str, bool]: Generated code and success status.
        """
        logger.debug(
            f"EnvAgent's tagged chat history - \n {format_tch(self.tagged_chat_history)}"
        )

        for attempt in range(max_attempts):
            logger.info(
                f"EnvAgent's in loop chat history - \n {format_tch(self.tagged_chat_history)}"
            )

            code, raw_response = gen_code(
                genner, to_normal_plist(self.tagged_chat_history)
            )

            ast_valid, ast_error = is_valid_code_ast(code)
            if not ast_valid:
                logger.error(
                    f"AST error - Attempt number {attempt + 1} - \n{ast_error}"
                )
                self.tagged_chat_history += [
                    ({"role": "assistant", "content": raw_response}, "gen_sp_egc(ast_fail)")
                ]
                self.tagged_chat_history += get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context=ast_error,
                    run_context="a Python AST compiler",
                )
                continue

            compile_valid, compiler_error = is_valid_code_compiler(code)
            if not compile_valid:
                logger.error(
                    f"Native `compile` error - Attempt number {attempt + 1} - \n{compiler_error}"
                )
                self.tagged_chat_history += [
                    ({"role": "assistant", "content": raw_response}, "gen_sp_egc(compile_fail)")
                ]
                self.tagged_chat_history += get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context=ast_error,
                    run_context="a Python native compiler",
                )
                continue

            exit_code, execution_output = run_code_in_con(
                docker_client, testing_container_id, code
            )
            if exit_code != 0:
                logger.error(
                    f"In container error - Attempt number {attempt + 1} - \n{execution_output}"
                )
                self.tagged_chat_history += [
                    ({"role": "assistant", "content": raw_response}, "gen_sp_egc(container_fail)")
                ]
                self.tagged_chat_history += get_code_regen_plist(
                    task_description="Generate and test code for getting non-basic environment info",
                    error_context=execution_output,
                    run_context="a Docker container",
                )
                continue

            self.tagged_chat_history += [
                ({"role": "assistant", "content": raw_response}, "gen_sp_egc(success)")
            ]

            return code, True

        return "", False

    def execute_sp_env_infos(
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
            _, container_result = run_code_in_con(docker_client, container_id, code)

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

    def save_tagged_chat_history_to_json(self, identifier: str, folder: Path | str):
        with open(
            Path(folder) / f"{identifier}_env_chat_history_latest.json", "w"
        ) as json_file:
            json.dump(self.tagged_chat_history, json_file, indent=4)


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
        basic_env_info_history: List[EnvironmentInfo],
        sp_env_info_history: List[str],
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
        self.old_tagged_chat_histories: List[List[TaggedMessage]] = []
        self.cur_strats: List[str] = []

        self.in_con_path = in_con_path

        # Add system prompt
        self.tagged_chat_history.extend(
            get_system_plist(str(in_con_path), str(basic_env_info_history))
        )
        # Add special env inclusion prompt
        self.tagged_chat_history.extend(
            get_special_env_plist(special_env_infos=sp_env_info_history)
        )
        # Add strategy gen prompt
        self.tagged_chat_history.extend(get_strat_req_plist(in_con_path=in_con_path))

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

    def gen_strats(self, genner: Callable) -> Tuple[List[str], str]:
        """Given an OAI client, will generate strategies based on the current chat history
        containing system prompt, initial environment info, and special environment info found by
        executing special environment info getters on a specified container.

        Args:
            oai_client: OpenAI client.

        Returns:
            strategies: List of string that are strategies.
        """

        return gen_list(genner, self.tagged_chat_history)

    def update_strat_state(self, strats: List[str], tag="strats_reply") -> str:
        """Given a newly generated strategies, will process the strategies so that it is appendable
        to the tagged chat history in the object.

        Args:
            strats (List[str]): Generated strategies.
            tag (str, optional): Tag of the processed strategy in tagged chat history. Defaults to "strats".

        Returns:
            processed_strats: Processed strat for debugging purposes.
        """

        self.cur_strats = strats

        strats_ = []

        for strat in strats:
            strats_.append("<Strat>")
            strats_.append(f"{strat}")
            strats_.append("</Strat>")

        processed_strats = "\n".join(strats_)

        self.tagged_chat_history.append(
            (
                {"role": "assistant", "content": processed_strats},  #
                tag,  #
            )
        )

        return processed_strats

    def update_env_info_state(
        self,
        fresh_basic_env_info: EnvironmentInfo,
        fresh_special_env_infos: List[str],
        basic_env_info_tag="get_basic_env_plist",
        special_env_info_tag="get_special_env_plist",
    ) -> int:
        # For saving and historical purposes
        self.old_tagged_chat_histories.append(self.tagged_chat_history)

        changes = 0

        for i in range(len(self.tagged_chat_history)):
            tag = self.tagged_chat_history[i][1]

            if tag == basic_env_info_tag:
                self.tagged_chat_history[i] = get_basic_env_plist(
                    str(fresh_basic_env_info)
                )[0]

                changes += 1

                continue
            elif tag == special_env_info_tag:
                self.tagged_chat_history[i] = get_special_env_plist(
                    fresh_special_env_infos
                )[0]

                changes += 1

                continue

        return changes

    def save_tagged_chat_history_to_json(self, identifier: str, folder: Path | str):
        # Save the old one
        for i, old_tagged_chat_history in enumerate(self.old_tagged_chat_histories):
            with open(
                Path(folder) / f"{identifier}_chat_history_{i}.json", "w"
            ) as json_file:
                json.dump(old_tagged_chat_history, json_file, indent=4)

        # Save the current one
        with open(
            Path(folder) / f"{identifier}_chat_history_latest.json", "w"
        ) as json_file:
            json.dump(self.tagged_chat_history, json_file, indent=4)
