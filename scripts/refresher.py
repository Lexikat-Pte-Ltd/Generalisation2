import sys
import tarfile
import time
from pathlib import Path
from typing import List, TypeGuard

import docker
from docker import DockerClient
from docker.models.containers import Container
from loguru import logger

from lib.container import safe_container_get

logger.add("logs/refresher.log")


def check_python_availability(client: DockerClient, container_id: str) -> bool:
    container = safe_container_get(client, container_id)

    exit_code, output = container.exec_run(cmd="python --version")

    python_available = exit_code == 0

    logger.debug(
        f"check_python_availability {client.api.base_url} {container_id[:10]} = {python_available}"
    )

    return python_available


def check_file_exists(client: DockerClient, container_id: str, file_path: str) -> bool:
    container = safe_container_get(client, container_id)
    exit_code, output = container.exec_run(cmd=f"ls {file_path}")

    file_exists = exit_code == 0

    logger.debug(
        f"check_file_exists {client.api.base_url} {container_id[:10]} {file_path} = {file_exists}"
    )

    return file_exists


def copy_to(client: DockerClient, container_id: str, src: str, dst: str):
    # Convert source and destination to Path objects
    src_path = Path(src)
    dst_path = Path(dst)

    # Create a tar file from the source
    tar_path = src_path.parent / (src_path.name + ".tar")

    with tar_path.open(mode="wb") as tar_file:
        with tarfile.open(fileobj=tar_file, mode="w") as tar:
            if src_path.is_file() or src_path.is_dir():
                tar.add(src_path, arcname=src_path.name)
            else:
                logger.error(
                    f"copy_to {client.api.base_url} {container_id[:10]} {src} {dst} - Source is neither file or dir"
                )
                raise ValueError("Source is neither file or dir")

    # Read the tar file's data
    data = tar_path.read_bytes()

    # Get the container
    container = safe_container_get(client, container_id)

    # Copy the tar file to the container
    is_successful = container.put_archive(str(dst_path.parent), data)

    if not is_successful:
        logger.error(
            f"copy_to {client.api.base_url} {container_id[:10]} {src} {dst} - Failed to untar the file inside the container."
        )
        raise ValueError("Failed to copy the file inside the container.")

    logger.debug(
        f"copy_to {client.api.base_url} {container_id[:10]} {src} {dst} - Copy and untar succeed"
    )

    # Optionally, clean up by removing the local tar file
    tar_path.unlink()


class Refresher:
    def __init__(
        self,
        client: DockerClient,
        container_ids: List[str],
        container_runner_path: str,
        local_runner_path: str,
    ):
        # On Host Config
        self.client = client
        self.container_ids = container_ids
        self.local_runner_path = local_runner_path

        # On Container Config
        self.container_runner_path = container_runner_path

    def check_and_refresh(self, container_id: str) -> bool:
        python_exists = check_python_availability(self.client, container_id)

        if not python_exists:
            logger.error(
                f"Refresher.refresh {container_id[:10]} - Python doesnt exists, raising error..."
            )
            raise ValueError("Python is missing")

        script_exists = check_file_exists(
            self.client, container_id, self.container_runner_path
        )

        if not script_exists:
            logger.info(
                f"Refresher.check_and_refresh - Files already exists on container {container_id[:10]}, skipping..."
            )

        logger.info("Refresher.check_and_refresh - Script doesnt exists, refreshing...")

        copy_to(
            self.client,
            container_id,
            src=self.local_runner_path,
            dst=self.container_runner_path,
        )

        refresh_success = check_file_exists(
            self.client, container_id, self.container_runner_path
        )

        logger.debug(
            f"Refresher.check_and_refresh {container_id[:10]} - Script doesnt exists, refreshing..."
        )

        return refresh_success

    def run(self, delay_time=1000):
        while True:
            for container_id in self.container_ids:
                self.check_and_refresh(container_id)
                time.sleep(delay_time)


if __name__ == "__main__":
    container_id = "befd8371212dc53f9d2e944b64dc6009bcb48658ca29418281b071514b4db964"
    container_script_path = "/runner.py"
    local_script_path = "./scripts/runner.py"

    client = docker.from_env()

    check_python_availability(client, container_id)
    check_file_exists(client, container_id, container_script_path)
    copy_to(client, container_id, local_script_path, container_script_path)
