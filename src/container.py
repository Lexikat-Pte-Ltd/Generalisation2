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
from src.helper import timeout
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


def get_memory_info_docker_stats(client: DockerClient, container_id: str) -> dict:
  """Get memory info using docker stats"""
  container = client.containers.get(container_id)
  stats = container.stats(stream=False)

  # Get memory stats in MB
  total_memory = stats["memory_stats"].get("limit", 0) // (1024 * 1024)
  used_memory = stats["memory_stats"].get("usage", 0) // (1024 * 1024)
  available_memory = total_memory - used_memory

  return {"total": total_memory, "available": available_memory, "used": used_memory}


def get_storage_info(container) -> Tuple[int, int]:
  """Get storage information from container"""
  try:
    exit_code, output = container.exec_run("df -m /")

    if exit_code == 0 and isinstance(output, bytes):
      lines = output.decode().splitlines()
      storage_info = lines[1].split()
      total = int(storage_info[1])  # MB
      available = int(storage_info[3])  # MB
      return total, available
    else:
      raise Exception("Failed to retrieve storage information caused by `df -m /`")
  except Exception as e:
    logger.debug(f"Storage info retrieval failed: {e}")
    raise e


def safe_detect_env(client: DockerClient, container_id: str) -> EnvironmentInfo:
  """Safely detect container environment info"""
  try:
    container = client.containers.get(container_id)

    # Get memory info
    mem_info = get_memory_info_docker_stats(client, container_id)

    # Get storage info
    total_storage, available_storage = get_storage_info(container)

    env_info = EnvironmentInfo(
      total_system_memory=mem_info["total"],
      available_system_memory=mem_info["available"],
      running_memory=mem_info["used"],
      total_storage=total_storage,
      available_storage=available_storage,
    )

    logger.debug(f"Container {container_id[:12]} environment info: {env_info}")
    return env_info
  except Exception as e:
    logger.error(f"Error detecting environment for container {container_id[:12]}: {e}")
    raise e


def write_code_in_con(
  container: Container, code: str, type: str, base_path: str = "/"
) -> Tuple[str, str]:
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
      base_path (str): The base path to write the code into
  Raises:
      Exception: If the file does not exist in the container

  Returns:
      str: The path to the temporary file in the container
      str: The reflected code
  """
  # Create temp file name with timestamp
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  temp_file_name = f"temp_script_{current_time}.py"
  temp_file_path = f"{base_path}/{temp_file_name}"

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
  succeed = container.put_archive(path=base_path, data=tar_stream.read())

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
  try:
    with timeout(seconds=150):
      python_exit_code, python_output = cast(
        Tuple[int, bytes],
        container.exec_run(
          cmd=command,
          demux=False,  # Combine stdout and stderr
          stream=False,  # Wait for the command to finish and return all output at once
        ),
      )
  except TimeoutError as e:
    container.exec_run(cmd="kill -9 $(pidof python)")
    return -1, str(e), ""
  except docker.errors.ContainerError as e:
    return -1, str(e), ""

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


def run_bash_in_con(container: Container, command: str) -> Tuple[int, str]:
  """Run bash command in container and return the exit code and output.

  Args:
    container (Container): The container to run the bash command in
    command (str): The bash command to run in the container

  Returns:
    int: The exit code
    str: The output
  """
  try:
    with timeout(seconds=150):
      exit_code, output = cast(
        Tuple[int, bytes],
        container.exec_run(cmd=command, demux=False, stream=False),
      )
  except TimeoutError as e:
    return -1, str(e)
  except docker.errors.ContainerError as e:
    return -1, str(e)

  if exit_code != 0:
    return (
      exit_code,
      f"Command {command} failed: {output.decode('utf-8', errors='replace')}",
    )

  return exit_code, output.decode("utf-8", errors="replace")


def repopulate_container(
  container: Container,
  base_path: Path | str = "./docker/fedora-learn-compose",
  username: str = "alice",
) -> None:
  """
  Restores files inside a running container from host files.
  Args:
      container: Docker container object
      base_path: Base path for source files
      username: Username in the container (defaults to 'alice')
  """
  try:
    # Set up paths
    base_path = Path(base_path)
    container_home = Path("/")

    # Define and process files
    files = {
      base_path / "tmp.tar.gz": Path("/"),
      base_path / "var.tar.gz": Path("/"),
      base_path / "documents.tar.gz": container_home,
      base_path / ".ssh": container_home / ".ssh",
      base_path / "scripts/randomly_encrypt.sh": container_home,
      base_path / "scripts/set_random_permissions.sh": container_home,
    }

    print(f"Restoring files to container {container.name}...")

    # Copy files to container
    for src, dest in files.items():
      if src.exists():
        print(f"Copying {src.name}...")
        with src.open("rb") as f:
          container.put_archive(str(dest), f.read())

    # Set permissions and run scripts
    for script in ["randomly_encrypt.sh", "set_random_permissions.sh"]:
      script_path = container_home / script
      container.exec_run(f"chmod 777 {script_path}")
      container.exec_run(f"/bin/bash {script_path}", workdir=str(container_home))

    print("Files restored successfully!")

  except Exception as e:
    print(f"Error during restoration: {str(e)}")
    raise
