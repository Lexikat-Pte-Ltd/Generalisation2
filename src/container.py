from datetime import datetime
import io
from pathlib import Path
import shlex
import tarfile
import tempfile
from typing import List, Literal, Tuple, cast

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


def parse_meminfo(meminfo_str: str) -> dict:
  """Parse /proc/meminfo content and convert values to MB."""
  mem_info = {}
  for line in meminfo_str.splitlines():
    if ":" not in line:
      continue

    key, value = line.split(":", 1)
    # Remove 'kB' and convert to MB (divide by 1024)
    value = value.strip()
    if "kB" in value:
      value = int(value.replace("kB", "").strip()) // 1024
    mem_info[key.strip()] = value
  return mem_info


def safe_detect_env(client: DockerClient, container_id: str) -> EnvironmentInfo:
  container = safe_container_get(client, container_id)

  # Get memory info from /proc/meminfo
  exit_code, output = container.exec_run(cmd="cat /proc/meminfo")

  if exit_code != 0 or not isinstance(output, bytes):
    logger.error(
      f"safe_detect_env {client.api.base_url} {container_id[:10]} - reading meminfo failed"
    )
    raise ValueError("Reading meminfo failed")

  mem_info = parse_meminfo(output.decode())

  # Get relevant memory values in MB
  total_system_memory = mem_info["MemTotal"]
  available_system_memory = mem_info["MemAvailable"]
  running_memory = total_system_memory - available_system_memory

  # Additional useful memory info for debugging
  logger.debug(f"Memory Info (MB):")
  logger.debug(f"  Total: {total_system_memory}")
  logger.debug(f"  Available: {available_system_memory}")
  logger.debug(f"  Used: {running_memory}")
  logger.debug(f"  Cached: {mem_info['Cached']}")
  logger.debug(f"  Buffers: {mem_info['Buffers']}")
  logger.debug(f"  Swap Total: {mem_info['SwapTotal']}")

  # Get storage information
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


def write_code_in_con(container: Container, code: str, type: str) -> Tuple[str, str]:
  """Write code into a temporary file in the host machine first then to the container.

  Algorithm:
  - Write code into a temporary file in the host machine
  - Create a tar archive containing the file
  - Copy the tar archive to the container's root directory
  - Check if the file exists in the container

  Args:
      container (Container): The container to write the code into
      code (str): The code to write into the container
      type (str): The type of the agent

  Raises:
      Exception: If the file does not exist in the container

  Returns:
      str: The path to the temporary file in the container
  """
  # Create temp file name with timestamp
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  temp_file_name = f"temp_script_{current_time}.py"

  # Create host file path and ensure directory exists
  logger.info(f"Writing file {temp_file_name} into host machine")
  host_path = Path(f"./data/temp_codes_{type}/{temp_file_name}")
  host_path.parent.mkdir(parents=True, exist_ok=True)
  host_path.write_text(code)

  # Create a tar archive in memory
  tar_stream = io.BytesIO()
  with tarfile.open(fileobj=tar_stream, mode="w") as tar:
    tar.add(host_path, arcname=temp_file_name)
  tar_stream.seek(0)

  # Copy the file to the container's root directory
  logger.info(f"Writing file {temp_file_name} into container")
  succeed = container.put_archive(path="/", data=tar_stream.read())

  if not succeed:
    raise Exception("Failed to write code into the container")

  # Check if file exists in container
  check_exist_command = (
    f"test -f /{temp_file_name} && echo 'File exists' || echo 'File does not exist'"
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
  reflected_code = container.exec_run(cmd=["cat", f"/{temp_file_name}"]).output.decode(
    "utf-8"
  )
  assert isinstance(reflected_code, str)

  return temp_file_name, reflected_code


def run_code_in_con(container: Container, code: str, type: str) -> Tuple[int, str, str]:
  """Run code in container and return the exit code, execution output, and reflected code.

  Algorithm:
  - Write code into a temporary file in the host machine
  - Create a tar archive containing the file
  - Copy the tar archive to the container's root directory
  - Check if the file exists in the container
  - Run the code in the container
  - Return the exit code, execution output, and reflected code

  Args:
    container (Container): The container to run the code in
    code (str): The code to run in the container
    type (str): The type of the agent

  Returns:
    int: The exit code
    str: The execution output
    str: The reflected code
  """
  temp_file_name, reflected_code = write_code_in_con(container, code, type)

  command = ["python", "-u", temp_file_name, "2>&1"]
  python_exit_code, python_output = cast(
    Tuple[int, bytes],
    container.exec_run(
      cmd=command,
      demux=False,  # Combine stdout and stderr
      stream=False,  # Wait for the command to finish and return all output at once
    ),
  )

  if python_exit_code != 0:
    logger.error(
      f"Run code in container failed: {python_output.decode('utf-8', errors='replace')}"
    )
    return python_exit_code, python_output.decode("utf-8", errors="replace"), ""

  return (
    python_exit_code,
    python_output.decode("utf-8", errors="replace"),
    reflected_code,
  )
