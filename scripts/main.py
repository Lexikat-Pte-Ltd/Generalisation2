import subprocess
from pathlib import Path
import sys
from typing import Dict, List, Optional

import cyclopts
import pydantic
from loguru import logger
from result import Err, Ok
from python_on_whales import Container as PowContainer, docker as pow_docker
from docker.models.containers import Container as DockerContainer

import docker
from config import config_from_toml
from src.agent import (
    generate_special_environment_getter_code,
    generate_strategy_list,
    regenerate_code,
    regenerate_list,
)
from src.genner import get_genner
from src.genner.Base import Genner
from src.helper import unflatten_toml_dict
from src.tool.code import validate_code_offline
from src.tool.docker import (
    get_container_free_disk_space,
    run_code_in_con,
    wait_and_get_container,
    write_code_in_con,
)
from src.typing.config import AppConfig
from src.typing.training import (
    SpecialEnvironmentGetterCodeTrainData,
    StrategyListTrainData,
)

app = cyclopts.App()


@app.command()
def main(config_file: str = "./config/multi-container.toml"):
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
    for container in containers:
        free_disk_space_byte, _ = get_container_free_disk_space(
            docker_client, container
        )

        env_info_dict[container.id] = (
            f"Container ID: {container.id}\n"
            f"Free Disk Space: {free_disk_space_byte} bytes\n"
        )

    env_infos: List[str] = list(env_info_dict.values())

    sp_env_infos, acc_sp_egc_train_data = special_environment_getter_code_flow(
        docker_client, containers, genner, config, env_infos
    )
    # TODO: SAVE SP EGC TRAIN DATA
    strategy_list, acc_sl_train_data = strategy_list_flow(
        genner, config, env_infos, sp_env_infos
    )


# TODO: SAVE SL TRAIN DATA


def special_environment_getter_code_flow(
    docker_client: docker.DockerClient,
    containers: List[DockerContainer],
    genner: Genner,
    config: AppConfig,
    env_infos: List[str],
):
    acc_sp_env_infos: List[str] = []
    acc_sp_egc_train_data: List[SpecialEnvironmentGetterCodeTrainData] = []
    current_attempt = 0

    should_regen = False
    error_sources: List[str] = []
    error_contexts: List[str] = []
    regen_count = 0
    latest_generation: Optional[str] = None

    while len(acc_sp_env_infos) < config.special_egc.count:
        if current_attempt > config.special_egc.max_retries:
            raise Exception(
                "Special environment getter code generation failed. Max retries exceeded, crashing on purpose."
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
            match generate_special_environment_getter_code(
                genner, env_infos, acc_sp_env_infos
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
                acc_sp_egc_train_data.append(
                    {
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "validation_result": error_message,
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
            host_cache_folder=Path(config.special_egc.code_host_cache_folder),
            code=code,
            postfix="",
            in_container_path="/",
        )

        match validate_code_offline(code):
            case Ok(_):
                pass
            case Err(error_message):
                acc_sp_egc_train_data.append(
                    {
                        "code": code,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "validation_result": error_message,
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
                acc_sp_egc_train_data.append(
                    {
                        "code": code,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "in_container_run_type": "ok",
                        "in_container_run_result": execution_output,
                    }
                )
                acc_sp_env_infos.append(execution_output)

                break
            case Err(error_message):
                acc_sp_egc_train_data.append(
                    {
                        "code": code,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                        "in_container_run_type": "err",
                        "in_container_run_result": error_message,
                    }
                )

                logger.error(f"Failed to run code in container: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)
                error_sources.append("code_run")

                continue

    logger.info("Special EGC stage completed.")
    logger.info(f"Special EGC train data: {acc_sp_egc_train_data}")
    logger.info(f"Special EGC env infos: {acc_sp_env_infos}")

    return acc_sp_env_infos, acc_sp_egc_train_data


def strategy_list_flow(
    genner: Genner,
    config: AppConfig,
    env_infos: List[str],
    special_env_infos: List[str],
):
    strategies: Optional[List[str]] = None
    acc_strategy_list_train_data: List[StrategyListTrainData] = []
    current_attempt = 0

    should_regen = False
    error_contexts: List[str] = []
    regen_count = 0
    latest_generation: Optional[str] = None

    while current_attempt > config.special_egc.max_retries:
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
                pass
            case Err(error_message):
                acc_strategy_list_train_data.append(
                    {
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "current_attempt": current_attempt,
                        "max_attempts": config.special_egc.max_retries,
                    }
                )
                logger.error(f"Failed to extract code: {error_message}")

                should_regen = True
                latest_generation = raw_response
                error_contexts.append(error_message)

                continue

    logger.info("Strategy list stage completed.")
    logger.info(f"Strategy list training data: {acc_strategy_list_train_data}")
    logger.info(f"Strategies generated: {strategies}")

    return strategies, acc_strategy_list_train_data


def strategy_code_flow():
    pass


if __name__ == "__main__":
    main()
