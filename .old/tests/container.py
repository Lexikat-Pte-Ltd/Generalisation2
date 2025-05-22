import subprocess
import time
import sys

from loguru import logger

import docker
from docker.errors import NotFound
from docker.models.containers import Container


def wait_for_container(container_name: str, timeout: int = 30) -> Container:
	"""Wait for container to be fully running and return the container object."""
	client = docker.from_env()
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

		except NotFound:
			logger.info(f"Container {container_name} not found yet, waiting...")
			time.sleep(1)
			continue

	raise TimeoutError(
		f"Container {container_name} did not start properly within {timeout} seconds"
	)


if __name__ == "__main__":
	while True:
		try:
			logger.info("Downing the container...")
			process = subprocess.Popen(
				"docker compose down",
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				bufsize=1,  # Line buffered
				shell=True,
				cwd="./docker/special-learn-compose",
			)
			logger.info("Container downing done...")

			return_code = process.wait()

			process = subprocess.Popen(
				"docker compose up -d",
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				bufsize=1,  # Line buffered
				shell=True,
				cwd="./docker/special-learn-compose",
			)

			assert process.stdout is not None
			assert process.stderr is not None

			return_code = process.wait()

			if return_code != 0:
				print(
					f"Docker compose up launching process exited with code {return_code}"
				)
				sys.exit(return_code)

			main_container_id = "special-learn-compose-service-a-1"
			test_container_id = "special-learn-compose-service-a-1"

			wait_for_container(main_container_id)


		except Exception as e:
			raise e
