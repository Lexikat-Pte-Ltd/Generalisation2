import io
import json
import shutil
import subprocess
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, cast

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
    container: DockerContainer, in_container_script_path: str
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
        container (DockerContainer): The container to run the code in
        temp_file_path (str): The path to the file in container containing the code

    Returns:
        Result[str, str]:
            - Ok: A string containing the execution_output
            - Err: An error message describing what went wrong

    Note:
        - The execution has a timeout of 150 seconds
        - After execution, any remaining Python processes are killed
    """
    rand_id = nanoid(12)

    command_str = (
        f"echo {rand_id};python -u {in_container_script_path} 2>&1"  # Use shell syntax
    )
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


def get_container_free_disk_space(client, container) -> Tuple[Optional[int], int]:
    container_size_rw = None
    container_size_root_fs = None  # To store SizeRootFs if found
    total_free_space_on_docker_root = -1

    print("--- Attempting to get SizeRw from container.attrs ---")
    # print(json.dumps(container.attrs, indent=2)) # Already printed in previous step

    try:
        # Initial attempt from container.attrs
        graph_driver_data = container.attrs.get("GraphDriver", {}).get("Data", {})
        if "SizeRw" in graph_driver_data and graph_driver_data["SizeRw"] is not None:
            container_size_rw = graph_driver_data["SizeRw"]
        elif container.attrs.get("SizeRw") is not None:
            container_size_rw = container.attrs["SizeRw"]

        if container_size_rw is not None:
            print(f"Found SizeRw in container.attrs: {container_size_rw}")
        else:
            print(
                "SizeRw not found in container.attrs. Attempting fallback using client.df()."
            )
            try:
                disk_usage_info = client.df()
                # print("\n--- client.df() output ---")
                # print(json.dumps(disk_usage_info, indent=2))
                # print("--- End of client.df() output ---\n")

                if "Containers" in disk_usage_info and disk_usage_info["Containers"]:
                    for c_info in disk_usage_info["Containers"]:
                        if c_info.get("Id") == container.id:
                            print(
                                f"Found container {container.id[:12]} in client.df() output."
                            )
                            print(
                                f"Container info from df(): {json.dumps(c_info, indent=2)}"
                            )
                            if c_info.get("SizeRw") is not None:
                                container_size_rw = c_info["SizeRw"]
                                print(
                                    f"Found SizeRw via client.df(): {container_size_rw}"
                                )
                            if c_info.get("SizeRootFs") is not None:
                                container_size_root_fs = c_info["SizeRootFs"]
                                print(
                                    f"Found SizeRootFs via client.df(): {container_size_root_fs}"
                                )
                            break  # Found our container
                    if container_size_rw is None and container_size_root_fs is None:
                        print(
                            f"Container {container.id[:12]} was found in client.df(), but SizeRw and SizeRootFs were not available in its entry."
                        )
                else:
                    print(
                        "No 'Containers' data found in client.df() output or it was empty."
                    )

            except Exception as e_df:
                print(f"Error attempting to use client.df(): {e_df}")

        if container_size_rw is None:
            print(
                "Could not determine SizeRw (writable layer size) through any method."
            )
            if container_size_root_fs is not None:
                print(
                    f"However, SizeRootFs (total image + writable layer) was found: {container_size_root_fs} bytes. This is a related metric."
                )

        # Get total free space on Docker root filesystem
        docker_root_dir = client.info().get("DockerRootDir")
        if docker_root_dir:
            try:
                usage = shutil.disk_usage(docker_root_dir)
                total_free_space_on_docker_root = usage.free
            except FileNotFoundError:
                print(
                    f"Error: DockerRootDir '{docker_root_dir}' not found. Cannot get disk usage."
                )
            except Exception as e_shutil:
                print(f"Error getting disk usage for '{docker_root_dir}': {e_shutil}")
        else:
            print("Could not determine DockerRootDir from client.info().")

    except Exception as e_main:
        print(f"An error occurred in get_container_free_disk_space: {e_main}")
        # Ensure tuple is returned matching signature
        return (container_size_rw if container_size_rw is not None else None), -1

    # Prioritize returning SizeRw if found.
    # If you want to return SizeRootFs when SizeRw is None, you'd adjust the logic here.
    # For now, the function signature implies returning SizeRw or None for the first element.
    return container_size_rw, total_free_space_on_docker_root


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
