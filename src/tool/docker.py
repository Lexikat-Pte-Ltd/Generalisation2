import io
import shutil
import subprocess
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, cast

from loguru import logger
from result import Err, Ok, Result

import docker
import docker.errors
from docker import DockerClient
from docker.errors import APIError as DockerAPIErrorException
from docker.errors import NotFound as DockerNotFoundException
from docker.models.containers import Container as DockerContainer
from src.helper import nanoid, timeout


def fetch_container(client: DockerClient, container_id: str):
    try:
        return client.containers.get(container_id)
    except docker.errors.NotFound as e:
        logger.error(
            "Container not found while trying to fetch container, \n"
            f"`container_id`: \n`{container_id}`,\n"
            f"`e`: \n`{e}`"
        )
        raise e
    except Exception as e:
        logger.error(
            "Unexpected error while trying to fetch container, \n"
            f"`container_id`: \n`{container_id}`,\n"
            f"`e`: \n`{e}`"
        )
        raise e


def write_code_in_con(
    client: DockerClient,
    container: DockerContainer,
    host_cache_folder: Path,
    code: str,
    postfix: str,
    in_container_path: str = "/",
) -> Tuple[str, str]:
    """Write code into a temporary file in the host machine first then to the container.

    Algorithm:
    - Write code into a temporary file in the host machine
    - Create a tar archive containing the file
    - Copy the tar archive to the container's root directory
    - Check if the file exists in the container

    Args:
        code (str): The code to write into the container
        postfix (str): The type identifier for the agent, used in the file path
        in_container_path (str, optional): The base path in the container to write the code to. Defaults to "/".

    Raises:
        Exception: If the file cannot be written to the container or if verification fails

    Returns:
        Tuple[str, str]:
            - The path to the temporary file in the container
            - The reflected code (content of the file as read from the container)
    """
    # Create temp file name with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file_name = f"temp_script_{current_time}.py"
    temp_file_path = f"{in_container_path}/{temp_file_name}"

    # Create host file path and ensure directory exists
    # logger.info(f"Writing file {temp_file_name} into host machine")
    host_path = host_cache_folder / f"temp_codes_{postfix}/{temp_file_name}"
    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_text(code)

    # Create a tar archive in memory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(host_path, arcname=temp_file_name)
    tar_stream.seek(0)

    # Copy the file to the container's root directory
    # logger.info(f"Writing file {temp_file_name} into container")

    succeed = container.put_archive(path=in_container_path, data=tar_stream.read())

    if not succeed:
        raise Exception("Failed to write code into the container")

    # Check if file exists in container
    check_exist_command = (
        f"test -f {temp_file_path} && echo 'File exists' || echo 'File does not exist'"
    )
    check_exist_result = container.exec_run(cmd=["/bin/sh", "-c", check_exist_command])

    if b"File exists" not in check_exist_result.output:
        logger.error(
            f"File verification failed: {check_exist_result.output.decode('utf-8')}"
        )
        raise Exception(
            f"File verification failed: {check_exist_result.output.decode('utf-8')}"
        )

    # Read the file content
    reflected_code = container.exec_run(cmd=["cat", temp_file_path]).output.decode(
        "utf-8"
    )
    assert isinstance(reflected_code, str)

    return temp_file_path, reflected_code


def run_code_in_con(
    container: DockerContainer, temp_file_path: str
) -> Result[str, str]:
    """Run code in container and return the exit code, execution output, and reflected code.

    Algorithm:
    - Write code into a temporary file in the host machine
    - Create a tar archive containing the file
    - Copy the tar archive to the container's root directory
    - Check if the file exists in the container
    - Run the code in the container
    - Return the exit code, execution output, and reflected code

    Args:
        code (str): The Python code to run in the container
        postfix (str): The type identifier for the agent, used in the file path

    Returns:
        Result[str, str]:
            - Ok: A string containing the execution_output
            - Err: An error message describing what went wrong

    Note:
        - The execution has a timeout of 150 seconds
        - After execution, any remaining Python processes are killed
    """
    rand_id = nanoid(12)

    command_str = f"echo {rand_id};python -u {temp_file_path} 2>&1"  # Use shell syntax
    cmd = ["/bin/sh", "-c", command_str]  # Execute via shell
    timeout_bool = False
    timeout_checker = time.time()

    def kill_processes_with_sudo(pid):
        try:
            # Run kill with sudo for processes not owned by the user
            subprocess.run(["sudo", "kill", "-9", str(pid)], check=True)
            print(f"Killed process {pid} using sudo")
        except subprocess.CalledProcessError as e:
            print(f"Failed to kill process {pid}: {e}")

    def kill_processes_by_name(keyword):
        try:
            # Run the ps command and filter with grep
            output = subprocess.check_output(["ps", "aux"], text=True)
            print(keyword)
            for line in output.splitlines():
                if keyword in line:
                    parts = line.split()
                    pid = int(parts[1])
                    print(f"Killing PID {pid}: {line}")
                    kill_processes_with_sudo(pid)

        except Exception as e:
            print(f"Error: {e}")

    timeout_seconds = 600
    try:
        with timeout(
            seconds=timeout_seconds,
            callback=lambda: kill_processes_by_name(rand_id),
        ):
            python_exit_code, python_output = cast(
                Tuple[int, bytes],
                container.exec_run(
                    cmd=cmd,
                    demux=False,  # Combine stdout and stderr
                    stream=False,  # Wait for the command to finish and return all output at once
                ),
            )
            if time.time() - timeout_checker > timeout_seconds:
                timeout_bool = True
            python_output_str = python_output.decode("utf-8", errors="replace")
    except TimeoutError as e:
        return Err(f"ContainerManager.run_code_in_con: Code ran too long, error: \n{e}")
    except docker.errors.ContainerError as e:
        return Err(f"ContainerManager.run_code_in_con: Container error, error: \n{e}")
    if timeout_bool:
        return Err(
            f"ContainerManager.run_code_in_con: Code ran too long, output: \n{python_output_str}"
        )

    container.exec_run(cmd="kill -9 $(pidof python)")

    if python_exit_code != 0:
        return Err(
            f"ContainerManager.run_code_in_con: Code that has been run failed, program output: \n{python_output_str}"
        )

    return Ok(python_output_str)


def get_container_free_disk_space(
    client: DockerClient, container: DockerContainer
) -> tuple[int, int]:
    """
    Gets the free disk space related to a Docker container.

    This function provides two pieces of information:
    1.  Size of the container's writable layer (how much it's currently using).
    2.  Available space on the filesystem hosting Docker's root directory.
        This is a general indicator of how much space is left for *all*
        Docker operations (new images, new container layers, etc.) on that
        partition, not just for this specific container. A container can
        theoretically grow until this space is exhausted.

    Args:
        container_id: The ID or name of the Docker container.

    Returns:
        A tuple containing:
            - int:
                Size of the container's writable layer in bytes.
            - int:
                Total free space in bytes on the Docker root filesystem.
    """
    try:
        # 1. Get the size of the container's writable layer
        container_size_rw = None
        if (
            "SizeRw" in container.attrs["GraphDriver"]["Data"]
        ):  # For overlay2, aufs etc.
            container_size_rw = container.attrs["GraphDriver"]["Data"]["SizeRw"]
        elif (
            container.attrs.get("SizeRw") is not None
        ):  # Fallback for some configurations
            container_size_rw = container.attrs["SizeRw"]

        # 2. Get free space on the Docker root directory's filesystem
        #    This is where the container's writable layer (and images) are stored.
        #    A container can grow until this space is consumed.
        docker_info = client.info()
        docker_root_dir = docker_info.get("DockerRootDir")
        host_free_space = None

        if docker_root_dir:
            try:
                # Get the partition/filesystem of the Docker root directory
                # Note: shutil.disk_usage gives usage for the *filesystem*
                # that docker_root_dir resides on.
                usage = shutil.disk_usage(docker_root_dir)
                host_free_space = usage.free
            except FileNotFoundError:
                logger.error(
                    f"Warning: Docker root directory '{docker_root_dir}' not found. "
                    "Cannot determine host free space."
                )
            except Exception as e:
                logger.error(
                    f"Warning: Could not get disk usage for '{docker_root_dir}': {e}"
                )
        else:
            logger.error(
                "Warning: Could not determine DockerRootDir to check host free space."
            )

        assert container_size_rw is not None, "Container size (RW) is None"
        assert host_free_space is not None, "Host free space is None"

        return container_size_rw, host_free_space

    except DockerNotFoundException as e:
        logger.error(
            "Docker container not found while getting container's free space, \n"
            f"`container.id`: {container.id}\n"
            f"`e`: \n{e}\n"
        )
        raise e
    except DockerAPIErrorException as e:
        logger.error(
            "Docker API error while getting container's free space, \n"
            f"`container.id`: {container.id}\n"
            f"`e`: \n{e}\n"
        )
        raise e
    except Exception as e:
        logger.error(
            "Unexpected error while getting container's free space, \n"
            f"`container.id`: {container.id}\n"
            f"`e`: \n{e}\n"
        )
        raise e


def wait_and_get_container(
    client: DockerClient, container_name: str, timeout: int = 30
) -> DockerContainer:
    """Wait for container to be fully running and return the container object."""
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
