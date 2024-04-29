from dataclasses import dataclass
from pathlib import Path
from typing import List

import docker
from docker import DockerClient
from pydantic import BaseModel
from loguru import logger

from lib.container import safe_container_get
from lib.data import EnvironmentInfo

logger.add("logs/controller.log")



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
    running_memory = available_system_memory // 2
    storage_space = available_system_memory - running_memory

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


def generate_task_list(
    client: DockerClient,
    container_id: str,
    in_con_path: str | Path,
    env_info: EnvironmentInfo,
    prompt_template: str,
    env_prompt_template: str,
):
    in_con_files = safe_filelist(client, container_id, in_con_path)
    env_prompt = env_prompt_template.format(
        env_info=env_info,
        in_con_files=in_con_files,
    )
    prompt = prompt_template.format(in_con_path=in_con_path)
    final_prompt = env_prompt + prompt
    


class AgentController:
    def __init__(
        self,
        client: DockerClient,
        container_id: str,
        in_con_path="/",
        out_con_path="./data/controller/",
    ):
        self.client = client
        self.container_id = container_id
        self.in_con_path = Path(in_con_path)  # For in container working path
        self.out_con_path = Path(out_con_path)  # For out container (host) working path

        self.logs_path = self.out_con_path / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.learned_skills_path = self.out_con_path / "learned_skils"
        self.learned_skills_path.mkdir(parents=True, exist_ok=True)
        self.train_backup_path = self.out_con_path / "learned_backup_path"
        self.train_backup_path.mkdir(parents=True, exist_ok=True)

        self.env_prompt = ""
        self.env_info = safe_detect_env(self.client, self.container_id)


if __name__ == "__main__":
    container_id = "befd8371212dc53f9d2e944b64dc6009bcb48658ca29418281b071514b4db964"

    client = docker.from_env()
    controller = AgentController(
        client=client,
        container_id=container_id,
    )
    print(controller.env_info)

    # safe_filelist(client, container_id, "/")
