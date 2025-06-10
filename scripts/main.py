import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import pydantic
from loguru import logger
from result import Err, Ok
from docker.models.containers import Container as DockerContainer

import docker
from config import config_from_toml
from src.agent import (
    generate_special_environment_getter_code,
    generate_strategy_code,
    generate_strategy_list,
    regenerate_code,
    regenerate_list,
)
from src.genner import get_genner
from src.genner.Base import Genner
from src.helper import (
    generate_readable_run_id,
    get_formatted_repo_info,
    string_hash,
    unflatten_toml_dict,
)
from src.tool.code import validate_code_offline
from src.tool.docker import (
    get_container_free_disk_space_kb_v1,
    get_container_free_disk_space_kb_v2,
    run_code_in_con,
    wait_and_get_container,
    write_code_in_con,
)
from src.typing.config import AppConfig
from src.typing.training import (
    SpecialEnvironmentGetterCodeTrainData,
    StrategyCodeTrainData,
    StrategyListTrainData,
    save_train_data,
)


RUN_ID = generate_readable_run_id()
COMMIT_ID = get_formatted_repo_info()


def main(config_file: str = "./config/multi-container.toml"):
    logger.info(f"Run ID: {RUN_ID}")
    logger.info(f"Commit ID: {COMMIT_ID}")

    config = config_from_toml(config_file, read_from_file=True)
    try:
        config = AppConfig(**unflatten_toml_dict(config.as_dict()))
    except pydantic.ValidationError as e:
        logger.info(f"Config validation error: {e}")
        return

    if config.dynamic_container:
        subprocess.run(["python", "scripts/container_launcher.py"])
        logger.info("Dynamic container launched")
    elif config.docker_compose_dir:
        logger.info("Downing the container...")
        process = subprocess.Popen(
            "docker compose down",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            shell=True,
            cwd=config.docker_compose_dir,
        )
        logger.info("Container downed...")

        assert process.stdout is not None
        assert process.stderr is not None

        return_code = process.wait()
        logger.info(f"Docker compose down exit code: {return_code}")

        logger.info("Relaunching container...")
        process = subprocess.Popen(
            "docker compose up -d --build",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            shell=True,
            cwd=config.docker_compose_dir,
        )
        logger.info("Container launched...")

        assert process.stdout is not None
        assert process.stderr is not None

        return_code = process.wait()

        if return_code != 0:
            print(f"Docker compose up launching process exited with code {return_code}")
            sys.exit(return_code)

    docker_client = docker.from_env()
    genner = get_genner("qwen")

    containers = []
    for container_id in config.container_ids:
        containers.append(wait_and_get_container(docker_client, container_id))

    env_info_dict: Dict[str, str] = {}
    containers_free_space: Dict[str, Tuple[Optional[float], float]] = {}
    for container in containers:
        assert container.id is not None

        rw_size_kb, free_space_v1_kb = get_container_free_disk_space_kb_v1(
            docker_client, container
        )

        if rw_size_kb is None or free_space_v1_kb is None:
            logger.warning(
                f"Failed to get read/write size or free space for container {container.id} using V1 method. "
            )

        free_space_v2_kb = get_container_free_disk_space_kb_v2(container)

        env_info = (
            f"Container ID: {container.id}\n"
            f"Read/Write Size: {rw_size_kb} KB\n"
            f"Free Disk Space (v1): {free_space_v1_kb} KB\n"
            f"Free Disk Space (v2): {free_space_v2_kb} KB\n"
        )
        logger.info(
            f"Env info for the container {container.id[:12]} is: \n{env_info.strip()}"
        )

        env_info_dict[container.id] = env_info
        containers_free_space[container.id] = (
            free_space_v1_kb,
            free_space_v2_kb,
        )

    # Convert the dictionary values to a list
    env_infos: List[str] = list(env_info_dict.values())

    sp_env_infos, sp_egc_train_data, sp_env_info_hashes = (
        special_environment_getter_code_flow(
            genner, docker_client, containers, config, env_infos
        )
    )
    save_train_data(
        "special_environment_getter_code",
        sp_egc_train_data,
        config.train_data_save_folder,
    )

    logger.info(f"Special environment infos: \n{sp_env_infos}")
    logger.info(f"`len(sp_egc_train_data)`: {len(sp_egc_train_data)}")

    strategies, strategy_list_train_data, strategies_hash = strategy_list_flow(
        genner, config, env_infos, sp_env_infos, sp_env_info_hashes
    )
    save_train_data(
        "strategy_list", strategy_list_train_data, config.train_data_save_folder
    )

    logger.info(f"Strategy list: \n{strategies}")
    logger.info(f"`len(strategy_list_train_data)`: {len(strategy_list_train_data)}")

    for strategy in strategies:
        strat_code, strat_code_hash, strat_code_train_data, space_freed_kb = (
            strategy_code_flow(
                genner,
                docker_client,
                containers,
                containers_free_space,
                config,
                strategy,
                strategies_hash,
                env_info_dict,
                sp_env_infos,
                sp_env_info_hashes,
            )
        )
        logger.info(f"Strategy code: \n{strat_code}")
        logger.info(f"Space freed: {space_freed_kb}")
        logger.info(f"`len(strat_code_train_data)`: {len(strat_code_train_data)}")
        save_train_data(
            "strategy_code",
            strat_code_train_data,
            config.train_data_save_folder,
        )


def special_environment_getter_code_flow(
    genner: Genner,
    docker_client: docker.DockerClient,
    containers: List[DockerContainer],
    config: AppConfig,
    env_infos: List[str],
):
    # Output variables
    sp_env_infos: List[str] = []
    sp_egc_train_data: List[SpecialEnvironmentGetterCodeTrainData] = []
    sp_env_info_hashes: List[str] = []

    # Loop variables
    current_attempt = 0
    regen_count = 0
    should_regen = False
    had_succeed = False
    error_sources: List[str] = []
    error_contexts: List[str] = []
    latest_generation: Optional[str] = None

    while len(sp_env_infos) < config.special_egc.count:
        if had_succeed:
            had_succeed = False
            error_sources: List[str] = []
            error_contexts: List[str] = []
            latest_generation: Optional[str] = None
            current_attempt = 0
            regen_count = 0

        if current_attempt > config.special_egc.max_retries:
            raise Exception(
                "Special environment getter code generation failed. Max retries exceeded, crashing on purpose."
            )

        logger.debug(f"Current attempt: {current_attempt}")
        current_attempt += 1

        if should_regen:
            logger.info(f"Regenning after {len(error_contexts)} mistakes for SP EGC...")

            should_regen = False
            assert latest_generation is not None

            match regenerate_code(
                genner, regen_count, error_sources, error_contexts, latest_generation
            ):
                case Ok((raw_response, prompt)):
                    logger.info(f"Regenerated a new code: \n{raw_response}")
                    pass
                case Err(error_message):
                    logger.error(f"Failed to regenerate code: \n{error_message}")
                    continue

            regen_count += 1
        else:
            logger.info("Generating new code for SP EGC...")
            match generate_special_environment_getter_code(
                genner, env_infos, sp_env_infos
            ):
                case Ok((raw_response, prompt)):
                    logger.info(f"Regenerated a new code: \n{raw_response}")
                    pass
                case Err(error_message):
                    logger.error(f"Failed to generate code: \n{error_message}")
                    continue
        logger.debug(f"Raw response generated: \n{raw_response}")

        logger.trace("Extracting code from raw response...")
        match genner.extract_code(raw_response):
            case Ok(code):
                pass
            case Err(error_message):
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_err": error_message,
                    }
                )
                logger.error(f"Failed to extract code. Error: \n{error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_extraction")

                continue

        logger.trace("Writing code in container...")
        in_container_code_path, code = write_code_in_con(
            docker_client,
            containers[config.main_container_idx],
            host_cache_folder=Path(config.code_host_cache_path) / "special_egc",
            code=code,
            postfix="",
            in_container_path="/",
        )
        logger.trace(
            f"Code written in container successfully with the name of {in_container_code_path}"
        )

        logger.trace("Validating code offline...")
        match validate_code_offline(code):
            case Ok(_):
                pass
            case Err(error_message):
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_ok": code,
                        "validation_result_if_err": error_message,
                    }
                )

                logger.error(f"Failed to validate code. Error: \n{error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_validation")

                continue

        logger.trace("Running code in container...")
        match run_code_in_con(
            containers[config.main_container_idx],
            in_container_code_path,
        ):
            case Ok(execution_output):
                if execution_output.strip() == "":
                    logger.error("Execution output is empty")

                    sp_egc_train_data.append(
                        {
                            "run_id": RUN_ID,
                            "version": COMMIT_ID,
                            "prompt": prompt,
                            "raw_response": raw_response,
                            "current_attempt": current_attempt,
                            "max_attempts": config.special_egc.max_retries,
                            "extract_result_if_ok": code,
                            "run_result_if_empty": "<nothing>",
                        }
                    )

                    should_regen = True
                    latest_generation = raw_response
                    error_contexts.append(
                        "Execution output is empty, please check the code."
                    )
                    error_sources.append("code_run")

                    continue

                execution_hash = string_hash(execution_output)
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_ok": code,
                        "run_result_if_ok": execution_output,
                        "run_result_if_ok_hash": execution_hash,
                    }
                )
                sp_env_infos.append(execution_output)
                sp_env_info_hashes.append(execution_hash)
                had_succeed = True

                logger.info(
                    f"Successfully ran code in container. Output: \n{execution_output.strip()}"
                )

            case Err(error_message):
                sp_egc_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "extract_result_if_ok": code,
                        "run_result_if_err": error_message,
                    }
                )

                logger.error(
                    f"Failed to run code in container. Error: \n{error_message}"
                )

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_run")

                continue

    logger.info("Special EGC stage completed.")
    logger.info(f"`len(sp_egc_train_data)`: {len(sp_egc_train_data)}")
    logger.info(f"`len(sp_env_infos)`: {len(sp_env_infos)}")

    for i, env_info in enumerate(sp_env_infos):
        logger.info(
            f"Special environment info {i + 1}: \n{env_info}\nHash: {sp_env_info_hashes[i]}"
        )

    return sp_env_infos, sp_egc_train_data, sp_env_info_hashes


def strategy_list_flow(
    genner: Genner,
    config: AppConfig,
    env_infos: List[str],
    special_env_infos: List[str],
    special_env_info_hashes: List[str],
):
    # Output variables
    strategies: Optional[List[str]] = None
    strategies_hash: Optional[str] = None
    strategy_list_train_data: List[StrategyListTrainData] = []

    # Loop variables
    current_attempt = 0
    regen_count = 0
    should_regen = False
    error_contexts: List[str] = []
    latest_generation: Optional[str] = None

    while strategies is None:
        if current_attempt > config.strategy_list.max_retries:
            raise Exception(
                "Strategy list generation failed. Max retries exceeded, crashing on purpose."
            )

        logger.debug(f"Current try: {current_attempt}")
        current_attempt += 1

        if should_regen:
            should_regen = False
            assert latest_generation is not None

            match regenerate_list(
                genner, regen_count, error_contexts, latest_generation
            ):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to regenerate list: {error_message}")
                    continue

            regen_count += 1
        else:
            match generate_strategy_list(genner, env_infos, special_env_infos, []):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to generate code: {error_message}")
                    continue

        match genner.extract_list(raw_response):
            case Ok(new_strategies):
                strategies = new_strategies
                strategies_hash = string_hash(",".join(strategies))

                strategy_list_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "special_env_info_hashes": special_env_info_hashes,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_list.max_retries,
                        "extract_result_if_ok": new_strategies,
                        "extract_result_if_ok_hash": strategies_hash,
                    }
                )
            case Err(error_message):
                strategy_list_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        "special_env_info_hashes": special_env_info_hashes,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_list.max_retries,
                        "extract_result_if_err": error_message,
                    }
                )
                logger.error(f"Failed to extract list: \n{error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)

                continue

    assert strategies is not None
    assert strategies_hash is not None

    logger.info("Strategy list stage completed.")
    logger.info(f"Strategy list training data: {strategy_list_train_data}")
    logger.info(f"Strategies generated: {strategies}")

    return strategies, strategy_list_train_data, strategies_hash


def strategy_code_flow(
    genner: Genner,
    docker_client: docker.DockerClient,
    containers: List[DockerContainer],
    containers_free_space: Dict[str, Tuple[Optional[float], float]],
    config: AppConfig,
    strategy: str,
    strategies_hash: str,
    env_info_dict: Dict[str, str],
    special_env_infos: List[str],
    special_env_info_hashes: List[str],
):
    # Output
    strat_code: Optional[str] = None
    strat_code_hash: Optional[str] = None
    strat_code_train_data: List[StrategyCodeTrainData] = []
    strat_space_freed_kb = 0

    # Loop variables
    current_attempt = 0
    regen_count = 0
    should_regen = False
    error_sources: List[str] = []
    error_contexts: List[str] = []
    latest_generation: Optional[str] = None

    while strat_space_freed_kb == 0:
        if current_attempt > config.strategy_code.max_retries:
            logger.error(
                "Strategy code generation failed. Max retries exceeded, crashing on purpose."
            )

        logger.debug(f"Cur retry: {current_attempt}")
        current_attempt += 1

        if should_regen:
            should_regen = False
            assert latest_generation is not None

            match regenerate_code(
                genner, regen_count, error_sources, error_contexts, latest_generation
            ):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to regenerate code: {error_message}")
                    continue

            regen_count += 1
        else:
            match generate_strategy_code(
                genner, strategy, list(env_info_dict.values()), special_env_infos
            ):
                case Ok((raw_response, prompt)):
                    pass
                case Err(error_message):
                    logger.error(f"Failed to generate code: {error_message}")
                    continue

        match genner.extract_code(raw_response):
            case Ok(code):
                pass
            case Err(error_message):
                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "extract_result_if_err": error_message,
                    }
                )
                logger.error(f"Failed to extract code: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_extraction")

                continue

        in_container_code_path, code = write_code_in_con(
            docker_client,
            containers[config.main_container_idx],
            host_cache_folder=Path(config.code_host_cache_path) / "strategy_code",
            code=code,
            postfix="",
            in_container_path="/",
        )

        match validate_code_offline(code):
            case Ok(_):
                pass
            case Err(error_message):
                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "validation_result_if_err": error_message,
                    }
                )

                logger.error(f"Failed to validate code: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_validation")

                continue

        match run_code_in_con(
            containers[config.main_container_idx],
            in_container_code_path,
        ):
            case Ok(execution_output):
                # Check if there's any space freed in V1 or V2
                containers_space_freed_kb = 0
                for container in containers:
                    assert container.id is not None

                    old_free_space_v1_kb, old_free_space_v2_kb = containers_free_space[
                        container.id
                    ]
                    new_rw_size_kb, new_free_space_v1_kb = (
                        get_container_free_disk_space_kb_v1(docker_client, container)
                    )
                    new_free_space_v2_kb = get_container_free_disk_space_kb_v2(
                        container
                    )

                    # Compare V1
                    space_freed_v1 = (
                        (new_free_space_v1_kb - old_free_space_v1_kb)
                        if old_free_space_v1_kb is not None
                        and new_free_space_v1_kb is not None
                        else None
                    )
                    # Compare V2
                    space_freed_v2 = new_free_space_v2_kb - old_free_space_v2_kb

                    logger.debug(f"V1 space freed is : {space_freed_v1} KB")
                    logger.debug(f"V2 space freed is : {space_freed_v2} KB")

                    containers_space_freed_kb += (
                        (space_freed_v1 + space_freed_v2) / 2
                        if space_freed_v1 is not None
                        else space_freed_v2
                    )

                    env_info_dict[container.id] = (
                        f"Container ID: {container.id}\n"
                        f"Read/Write Size: {new_rw_size_kb} KB\n"
                        f"Free Disk Space (v1): {new_free_space_v1_kb} KB\n"
                        f"Free Disk Space (v2): {new_free_space_v2_kb} KB\n"
                    )
                    containers_free_space[container.id] = (
                        new_free_space_v1_kb,
                        new_free_space_v2_kb,
                    )

                strat_code = code
                strat_code_hash = string_hash(code)
                strat_space_freed_kb += containers_space_freed_kb

                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "extract_result_if_ok": code,
                        "run_result_if_ok": execution_output,
                        "run_result_if_ok_hash": strat_code_hash,
                        "this_code_space_change_kb": containers_space_freed_kb,
                    }
                )

                break
            case Err(error_message):
                strat_code_train_data.append(
                    {
                        "run_id": RUN_ID,
                        "version": COMMIT_ID,
                        #
                        "strategies_hash": strategies_hash,
                        "special_env_info_hashes": special_env_info_hashes,
                        #
                        "strategy": strategy,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.strategy_code.max_retries,
                        #
                        "extract_result_if_ok": code,
                        "run_result_if_err": error_message,
                    }
                )

                logger.error(f"Failed to run code in container: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_run")

                continue

    logger.info(f"Strategy code completed on strat \n{strategy}.")
    logger.info(f"Strategy code train data: {strat_code_train_data}")
    logger.info(f"Strategy code: {strat_code}")
    logger.info(f"Total space freed: {strat_space_freed_kb} KB")

    return (
        strat_code,
        strat_code_hash,
        strat_code_train_data,
        strat_space_freed_kb,
    )


if __name__ == "__main__":
    main()
