from pathlib import Path
import shlex
import tempfile
from typing import List, Tuple, cast

import docker.errors
from loguru import logger

import docker
from docker import DockerClient
from docker.models.containers import Container
from src.data import EnvironmentInfo


def safe_container_get(client: DockerClient, container_identifier: str) -> Container:
    container = client.containers.get(container_identifier)

    try:
        container = client.containers.get(container_identifier)
    except docker.errors.NotFound:
        # If not found, try listing all containers and searching by name
        all_containers = client.containers.list(all=True)
        matching_containers = [
            c for c in all_containers if container_identifier in (c.name, c.id)
        ]
        if not matching_containers:
            logger.error(f"Container not found: {container_identifier}")
            raise ValueError("Container not found")
        container = matching_containers[0]

    if not isinstance(container, Container):
        logger.error(f"Retrieved object is not a Container: {container_identifier}")
        raise ValueError("Retrieved object is not a Container")

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
) -> Tuple[int, str, str]:
    container = safe_container_get(client, container_id)

    temp_file = f"/temp_script_{tempfile.NamedTemporaryFile().name.split('/')[-1]}.py"

    try:
        # Escape the code for shell
        escaped_shell_code = shlex.quote(escaped_code)

        # Use echo instead of heredoc
        write_command = f"echo {escaped_shell_code} > {temp_file}"
        write_result = container.exec_run(cmd=["/bin/sh", "-c", write_command])
        logger.info(f"Writing into file {temp_file}")

        if write_result.exit_code != 0:
            raise Exception(
                write_result.exit_code,
                f"Failed to create file: {write_result.output.decode('utf-8')}",
            )

        verify_command = (
            f"test -f {temp_file} && echo 'File exists' || echo 'File does not exist'"
        )
        verify_result = container.exec_run(cmd=["/bin/sh", "-c", verify_command])

        if b"File exists" not in verify_result.output:
            raise Exception(
                1, f"File verification failed: {verify_result.output.decode('utf-8')}"
            )

        command = ["python", "-u", temp_file, "2>&1"]

        result = cast(
            Tuple[int, bytes],
            container.exec_run(
                cmd=command,
                demux=False,  # Combine stdout and stderr
                stream=False,  # Wait for the command to finish and return all output at once
            ),
        )

        exit_code, output = result

        # Read the contents of the temporary file
        cat_command = f"cat {temp_file}"
        cat_result = container.exec_run(cmd=["/bin/sh", "-c", cat_command])
        file_contents = cat_result.output.decode("utf-8", errors="replace")

        return exit_code, output.decode("utf-8", errors="replace"), file_contents
    finally:
        # container.exec_run(cmd=["/bin/sh", "-c", f"rm -f {temp_file}"])
        pass
