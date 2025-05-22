import subprocess
from typing import Dict

import cyclopts

import docker
from config import config_from_toml
from loguru import logger
from src.tool.docker import get_container_free_disk_space, wait_and_get_container
from src.typing.config import AppSettings

app = cyclopts.App()


@app.command()
def main(config_file: str = "config/multi-container.toml"):
    config = config_from_toml(config_file, read_from_file=True)
    config = AppSettings(**config.as_dict())

    if config.dynamic_container:
        subprocess.run(["python", "scripts/container_launcher.py"])
        logger.info("Dynamic container launched")

    docker_client = docker.from_env()

    containers = []
    for container_id in config.container_ids:
        containers.append(wait_and_get_container(docker_client, container_id))

    env_infos: Dict[str, str] = {}
    for container in containers:
        free_disk_space_byte, _ = get_container_free_disk_space(
            docker_client, container
        )

        env_infos[container.id] = (
            f"Container ID: {container.id}\n"
            f"Free Disk Space: {free_disk_space_byte} bytes\n"
        )
    


if __name__ == "__main__":
    main()
