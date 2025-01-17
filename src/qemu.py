from datetime import datetime
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from loguru import logger


@dataclass
class QEMUEnvironmentInfo:
  total_system_memory: int  # in MB
  available_system_memory: int  # in MB
  running_memory: int  # in MB
  total_storage: int  # in MB
  available_storage: int  # in MB


class QEMUVirtualMachine:
  def __init__(self, vm_name: str, monitor_socket: str):
    self.vm_name = vm_name
    self.monitor_socket = monitor_socket
    self._validate_monitor()

  def _validate_monitor(self) -> None:
    """Validate that the QEMU monitor socket exists and is accessible."""
    if not os.path.exists(self.monitor_socket):
      logger.error(f"Monitor socket not found: {self.monitor_socket}")
      raise ValueError("Monitor socket not found")

  def _send_monitor_command(self, command: str) -> Dict[str, Any]:
    """Send a command to QEMU monitor and return the response."""
    try:
      # Using socat to communicate with QEMU monitor socket
      cmd = f"echo '{command}' | socat - UNIX-CONNECT:{self.monitor_socket}"
      result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=True
      )
      return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
      logger.error(f"Monitor command failed: {e}")
      raise ValueError(f"Monitor command failed: {e}")
    except json.JSONDecodeError as e:
      logger.error(f"Failed to parse monitor response: {e}")
      raise ValueError(f"Failed to parse monitor response: {e}")


def safe_vm_get(vm_name: str, monitor_socket: str) -> QEMUVirtualMachine:
  """Safely get a QEMU VM instance."""
  try:
    vm = QEMUVirtualMachine(vm_name, monitor_socket)
    return vm
  except ValueError as e:
    # If not found, try listing all VMs and searching by name
    all_vms = subprocess.run(
      ["ps", "-ef", "|", "grep", "qemu"], capture_output=True, text=True
    )
    if vm_name not in all_vms.stdout:
      logger.error(f"VM not found: {vm_name}")
      raise ValueError("VM not found")
    raise


def safe_folderlist_bfs(
  vm: QEMUVirtualMachine,
  path: str | Path,
  max_depth=3,
  ssh_config: Optional[Dict[str, str]] = None,
) -> List[str]:
  """Get folder list from VM using BFS approach."""
  folders: List[str] = []

  ssh_cmd = "ssh"
  if ssh_config:
    ssh_cmd += f" -i {ssh_config['key_path']} {ssh_config['user']}@{ssh_config['host']}"

  for i in range(1, max_depth + 1):
    bash_command = f"find {path} -mindepth {i} -maxdepth {i} -type d"
    try:
      result = subprocess.run(
        f"{ssh_cmd} '{bash_command}'",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
      )
      folders.extend(result.stdout.splitlines())
    except subprocess.CalledProcessError as e:
      logger.error(f"Failed to list folders: {e}")
      raise ValueError(f"Failed to list folders: {e}")

  return folders


def safe_filelist(
  vm: QEMUVirtualMachine,
  path: str | Path,
  shallow=True,
  ssh_config: Optional[Dict[str, str]] = None,
) -> List[str]:
  """Get file list from VM."""
  ssh_cmd = "ssh"
  if ssh_config:
    ssh_cmd += f" -i {ssh_config['key_path']} {ssh_config['user']}@{ssh_config['host']}"

  bash_command = (
    f"find {path} -maxdepth 2 -type f" if shallow else f"find {path} -type f"
  )

  try:
    result = subprocess.run(
      f"{ssh_cmd} '{bash_command}'",
      shell=True,
      capture_output=True,
      text=True,
      check=True,
    )
    return result.stdout.splitlines()
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to list files: {e}")
    raise ValueError(f"Failed to list files: {e}")


def safe_detect_env(
  vm: QEMUVirtualMachine, ssh_config: Optional[Dict[str, str]] = None
) -> QEMUEnvironmentInfo:
  """Detect environment information from VM."""
  ssh_cmd = "ssh"
  if ssh_config:
    ssh_cmd += f" -i {ssh_config['key_path']} {ssh_config['user']}@{ssh_config['host']}"

  try:
    # Get memory info
    meminfo_cmd = f"{ssh_cmd} 'cat /proc/meminfo'"
    meminfo_result = subprocess.run(
      meminfo_cmd, shell=True, capture_output=True, text=True, check=True
    )
    mem_info = parse_meminfo(meminfo_result.stdout)

    # Get storage info
    storage_cmd = f"{ssh_cmd} 'df -m /'"
    storage_result = subprocess.run(
      storage_cmd, shell=True, capture_output=True, text=True, check=True
    )

    storage_lines = storage_result.stdout.splitlines()
    storage_info = storage_lines[1].split()

    return QEMUEnvironmentInfo(
      total_system_memory=mem_info["MemTotal"],
      available_system_memory=mem_info["MemAvailable"],
      running_memory=mem_info["MemTotal"] - mem_info["MemAvailable"],
      total_storage=int(storage_info[1]),
      available_storage=int(storage_info[3]),
    )
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to detect environment: {e}")
    raise ValueError(f"Failed to detect environment: {e}")


def write_code_in_vm(
  vm: QEMUVirtualMachine, code: str, type: str, ssh_config: Dict[str, str]
) -> Tuple[str, str]:
  """Write code into VM using SSH."""
  # Create temp file name with timestamp
  current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  temp_file_name = f"temp_script_{current_time}.py"

  # Create temporary file locally
  with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
    temp_file.write(code)
    local_path = temp_file.name

  try:
    # Copy file to VM using scp
    scp_cmd = f"scp -i {ssh_config['key_path']} {local_path} {ssh_config['user']}@{ssh_config['host']}:/{temp_file_name}"
    subprocess.run(scp_cmd, shell=True, check=True)

    # Verify file exists and get content
    ssh_cmd = (
      f"ssh -i {ssh_config['key_path']} {ssh_config['user']}@{ssh_config['host']}"
    )
    verify_cmd = f"{ssh_cmd} 'cat /{temp_file_name}'"
    result = subprocess.run(
      verify_cmd, shell=True, capture_output=True, text=True, check=True
    )

    return temp_file_name, result.stdout
  finally:
    # Clean up local temporary file
    os.unlink(local_path)


def run_code_in_vm(
  vm: QEMUVirtualMachine, code: str, type: str, ssh_config: Dict[str, str]
) -> Tuple[int, str, str]:
  """Run code in VM and return results."""
  temp_file_name, reflected_code = write_code_in_vm(vm, code, type, ssh_config)

  try:
    # Run the code in VM
    ssh_cmd = (
      f"ssh -i {ssh_config['key_path']} {ssh_config['user']}@{ssh_config['host']}"
    )
    run_cmd = f"{ssh_cmd} 'python3 -u /{temp_file_name} 2>&1'"
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)

    return result.returncode, result.stdout, reflected_code
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to run code: {e}")
    return 1, str(e), reflected_code


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
