from pathlib import Path
from typing import List, Tuple, cast

from docker import DockerClient
from docker.models.containers import Container
from loguru import logger

from src.data import EnvironmentInfo


def safe_container_get(client: DockerClient, container_id: str) -> Container:
    container = client.containers.get(container_id)

    if not isinstance(container, Container):
        logger.error(
            f"safe_container_get {client.api.base_url} {container_id[:10]} - Container not found"
        )
        raise ValueError("Container not found")

    return container


def safe_folderlist_bfs(
    client: DockerClient, container_id: str, path: str | Path, max_depth=3
) -> List[str]:
    container = safe_container_get(client, container_id)
    folders: List[str] = []

    for i in range(1, max_depth + 1):
        bash_command = f"find -mindepth {i} -maxdepth {i} -type d"
        exit_code, output = container.exec_run(cmd=bash_command)

        if exit_code != 0 or not isinstance(output, bytes):
            logger.error(
                f"safe_get_filelist {client.api.base_url} {container_id[:10]} - `{bash_command}` failed"
            )
            raise ValueError(f"`{bash_command}` failed")

        lines = output.decode().splitlines()

        folders = folders + lines

    return folders


def safe_filelist(
    client: DockerClient, container_id: str, path: str | Path, shallow=True
) -> List[str]:
    container = safe_container_get(client, container_id)

    bash_command = (
        f"find {path} -maxdepth 2 -type f" if shallow else f"find {path} -type f"
    )
    exit_code, output = container.exec_run(cmd=bash_command)

    if exit_code != 0 or not isinstance(output, bytes):
        logger.error(
            f"safe_get_filelist {client.api.base_url} {container_id[:10]} - `{bash_command}` failed"
        )
        raise ValueError(f"`{bash_command}` failed")

    lines = output.decode().splitlines()

    return lines


def safe_detect_env(client: DockerClient, container_id: str) -> EnvironmentInfo:
    container = safe_container_get(client, container_id)

    # Calculate running and storage space based on available memory
    exit_code, output = container.exec_run(cmd="free -m")

    if exit_code != 0 or not isinstance(output, bytes):
        logger.error(
            f"safe_detect_env {client.api.base_url} {container_id[:10]} - `free -m` failed"
        )
        raise ValueError("`free -m` failed")

    lines = output.decode().splitlines()
    mem_info = lines[1].split()

    total_system_memory = int(mem_info[1])
    available_system_memory = int(mem_info[6])
    running_memory = total_system_memory - available_system_memory

    # Get storage (disk space) information
    exit_code, output = container.exec_run(cmd="df -m /")

    if exit_code != 0 or not isinstance(output, bytes):
        logger.error(
            f"safe_detect_env {client.api.base_url} {container_id[:10]} - `df -m /` failed"
        )
        raise ValueError("`df -m /` failed")

    lines = output.decode().splitlines()
    storage_info = lines[1].split()

    total_storage = int(storage_info[1])
    available_storage = int(storage_info[3])

    env_info = EnvironmentInfo(
        # From `free -m`
        total_system_memory=total_system_memory,
        available_system_memory=available_system_memory,
        running_memory=running_memory,
        # From `df -m /`
        total_storage=total_storage,
        available_storage=available_storage,
    )

    logger.debug(
        f"safe_detect_env {client.api.base_url} {container_id[:10]} = {env_info}"
    )

    return env_info


def run_code_in_con(
    client: DockerClient, container_id: str, escaped_code: str
) -> Tuple[int, str]:
    container = safe_container_get(client, container_id)

    command = f"python -c \"{escaped_code}\""

    # Calculate running and storage space based on available memory
    result = cast(Tuple[int, bytes], container.exec_run(cmd=command))
    exit_code, output = result

    return exit_code, output.decode()

