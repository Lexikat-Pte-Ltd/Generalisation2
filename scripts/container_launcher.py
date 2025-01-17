import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

import docker
import docker.errors

os.environ["TZ"] = "Asia/Jakarta"  # or 'Asia/Singapore'

skips = [
  "rocket.chat:7.2.0",
  "eclipse-mosquitto:2.0.20",
  "opencart:latest",
  "grafana:9.3.6",
  "sonarqube:9-community",
  "minio:RELEASE.2023-01-25T00-19-54Z",
  "arangodb:3.9",
]

DOCKER_IMAGES = [
  {
    "name": "nginx:alpine",
    "description": "Lightweight web server",
    "category": "web_server",
    "base_os": "alpine",
    "variants": ["debian", "alpine"],
  },
  {
    "name": "postgres:14-alpine",
    "description": "PostgreSQL database system",
    "category": "database",
    "base_os": "alpine",
    "variants": ["debian", "alpine"],
  },
  {
    "name": "adoptopenjdk:16-jdk-hotspot",
    "description": "Java development kit",
    "category": "development",
    "base_os": "debian",
    "variants": ["alpine", "ubuntu", "debian", "centos"],
  },
  {
    "name": "redis:6.2",
    "description": "In-memory data structure store",
    "category": "database",
    "base_os": "debian",
    "variants": ["alpine", "debian"],
  },
  {
    "name": "rust:1.68",
    "description": "Rust programming environment",
    "category": "development",
    "base_os": "debian",
    "variants": ["alpine", "debian", "slim"],
  },
  {
    "name": "php:8.1-fpm",
    "description": "PHP FastCGI implementation",
    "category": "development",
    "base_os": "debian",
    "variants": ["alpine", "debian", "zts"],
  },
  {
    "name": "neo4j:4.4",
    "description": "Graph database platform",
    "category": "database",
    "base_os": "debian",
    "variants": ["debian", "ubuntu"],
  },
  {
    "name": "rabbitmq:3.9-management",
    "description": "Message broker with management interface",
    "category": "messaging",
    "base_os": "debian",
    "variants": ["alpine", "debian", "ubuntu"],
  },
  {
    "name": "ghost:4-alpine",
    "description": "Blogging platform",
    "category": "cms",
    "base_os": "alpine",
    "variants": ["alpine", "debian"],
  },
  {
    "name": "mariadb:10.8",
    "description": "MySQL fork database",
    "category": "database",
    "base_os": "ubuntu",
    "variants": ["ubuntu", "debian"],
  },
  {
    "name": "drupal:9",
    "description": "Content management system",
    "category": "cms",
    "base_os": "debian",
    "variants": ["apache", "fpm", "fpm-alpine"],
  },
  {
    "name": "gradle:7.6-jdk17",
    "description": "Build automation tool",
    "category": "build",
    "base_os": "debian",
    "variants": ["jdk", "alpine"],
  },
  {
    "name": "wordpress:6-apache",
    "description": "Blog and CMS platform",
    "category": "cms",
    "base_os": "debian",
    "variants": ["apache", "fpm", "fpm-alpine"],
  },
  {
    "name": "cassandra:4.0",
    "description": "NoSQL database",
    "category": "database",
    "base_os": "debian",
    "variants": ["debian", "jdk"],
  },
  {
    "name": "vault:1.12.0",
    "description": "Secrets management tool",
    "category": "security",
    "base_os": "alpine",
    "variants": ["alpine", "debian"],
  },
  {
    "name": "rocket.chat:7.2.0",
    "description": "Team chat platform",
    "category": "communication",
    "base_os": "ubuntu",
    "variants": ["ubuntu"],
  },
  {
    "name": "caddy:2.9.1-alpine",
    "description": "Web server with automatic HTTPS",
    "category": "web_server",
    "base_os": "alpine",
    "variants": ["alpine", "debian", "windows"],
  },
  {
    "name": "eclipse-mosquitto:2.0.20",
    "description": "MQTT message broker",
    "category": "messaging",
    "base_os": "alpine",
    "variants": ["alpine", "debian"],
  },
  {
    "name": "perl:devel",
    "description": "Perl programming environment",
    "category": "development",
    "base_os": "debian",
    "variants": ["threaded", "slim", "alpine"],
  },
  {
    "name": "opencart:latest",
    "description": "E-commerce platform",
    "category": "ecommerce",
    "base_os": "debian",
    "variants": ["apache", "fpm"],
  },
  {
    "name": "grafana:9.3.6",
    "description": "Analytics and monitoring platform",
    "category": "monitoring",
    "base_os": "ubuntu",
    "variants": ["ubuntu", "alpine"],
  },
  {
    "name": "sonarqube:9-community",
    "description": "Code quality platform",
    "category": "quality",
    "base_os": "alpine",
    "variants": ["alpine", "debian"],
  },
  {
    "name": "minio:RELEASE.2023-01-25T00-19-54Z",
    "description": "Object storage server",
    "category": "storage",
    "base_os": "alpine",
    "variants": ["alpine", "debian", "windows"],
  },
  {
    "name": "arangodb:3.9",
    "description": "Multi-model database",
    "category": "database",
    "base_os": "ubuntu",
    "variants": ["ubuntu"],
  },
]

logger.add("logs/container_launcher.log")


class DockerHandler:
  def __init__(
    self,
    source_dir: Path | str,
    dst_dir: Path | str,
  ):
    self.source_dir = Path(source_dir)
    self.dst_dir = Path(dst_dir)
    self.dockerfile_content = []
    self.detected_os = None
    logger.info(f"Source dir: {self.source_dir}")
    logger.info(f"Destination dir: {self.dst_dir}")

  def prepare_build_context(self) -> None:
    """Prepare the build context by copying required files."""
    self.dst_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating directory {self.dst_dir}")

    # Write the permissions script with explicit syntax
    permissions_script = self.dst_dir / "set_random_permissions.sh"
    with open(permissions_script, "w", newline="\n") as f:
      f.write("#!/bin/sh\n")
      f.write("for file in *; do\n")
      f.write('  if [ -f "$file" ]; then\n')
      f.write('    chmod $(( RANDOM % 777 + 1 )) "$file"\n')
      f.write("  fi\n")
      f.write("done\n")
    permissions_script.chmod(0o755)

    # Write the encrypt script with explicit syntax
    encrypt_script = self.dst_dir / "randomly_encrypt.sh"
    with open(encrypt_script, "w", newline="\n") as f:
      f.write("#!/bin/sh\n")
      f.write("for file in documents.tar.gz tmp.tar.gz var.tar.gz; do\n")
      f.write('  if [ -f "$file" ]; then\n')
      f.write(
        '    openssl enc -aes-256-cbc -salt -in "$file" -out "$file.enc.tmp" -k "defaultpassword"\n'
      )
      f.write('    mv "$file.enc.tmp" "$file"\n')
      f.write("  fi\n")
      f.write("done\n")
    encrypt_script.chmod(0o755)

    # Handle other required files
    required_files = [
      "documents.tar.gz",
      "tmp.tar.gz",
      "var.tar.gz",
      ".ssh",
    ]

    for file in required_files:
      src = self.source_dir / file
      dst = self.dst_dir / file

      logger.info(f"Copying {src} to {dst}")

      if src.exists():
        if src.is_dir():
          shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
          shutil.copy2(src, dst)
      else:
        raise FileNotFoundError(f"Required file not found: {src}")

  def detect_base_os(self, base_os: str) -> str:
    """Detect the base OS from the image name."""
    base_os_lower = base_os.lower()
    if "alpine" in base_os_lower:
      return "alpine"
    elif "fedora" in base_os_lower:
      return "fedora"
    elif "debian" in base_os_lower or "ubuntu" in base_os_lower:
      return "debian"
    elif "centos" in base_os_lower:
      return "centos"
    else:
      raise ValueError(f"Unsupported base OS in image: {base_os}")

  def generate_dockerfile(self, base_image: str, base_os: str, username: str) -> None:
    """Generate Dockerfile content based on OS and configuration."""
    self.detected_os = self.detect_base_os(base_os)
    logger.info(f"Detected OS: {self.detected_os}")

    # Start with base image
    self.dockerfile_content = [f"FROM {base_image}"]

    # Add OS-specific package installation for Python and venv support
    if self.detected_os == "alpine":
      self.dockerfile_content.append(
        "RUN apk update && "
        "apk add --no-cache "
        "openssh "
        "openssl "
        "python3 "
        "python3-dev "  # For compiling Python packages
        "py3-pip "
        "py3-virtualenv "
        "gcc "
        "musl-dev "  # Required for compiling
        "linux-headers "  # Required for psutil
        "bash "
        "procps "
        "coreutils "
        "util-linux"
      )
    elif self.detected_os == "debian":
      self.dockerfile_content.append(
        "RUN apt-get update && "
        "apt-get install -y "
        "bash "
        "openssl "
        "openssh-server "
        "openssh-client "
        "python3 "
        "python3-pip "
        "python3-venv "
        "python3-dev "  # For compiling Python packages
        "gcc "
        "linux-headers-generic "  # Required for psutil
        "procps "
        "coreutils "
        "util-linux && "
        "rm -rf /var/lib/apt/lists/*"
      )
    elif self.detected_os == "fedora":
      self.dockerfile_content.append(
        "RUN dnf update -y && "
        "dnf install -y "
        "bash "
        "openssl "
        "openssh-server "
        "openssh-clients "
        "python3 "
        "python3-pip "
        "python3-virtualenv "
        "python3-devel "  # For compiling Python packages
        "gcc "
        "kernel-headers "  # Required for psutil
        "procps-ng "
        "coreutils "
        "util-linux"
      )
    elif self.detected_os == "centos":
      self.dockerfile_content.append(
        "RUN yum update -y && "
        "yum install -y "
        "bash "
        "openssl "
        "openssh-server "
        "openssh-clients "
        "python3 "
        "python3-pip "
        "python3-devel "  # For compiling Python packages
        "gcc "
        "kernel-headers "  # Required for psutil
        "procps "
        "coreutils "
        "util-linux"
      )

    # Add user creation
    if self.detected_os == "alpine":
      self.dockerfile_content.append(
        f"RUN adduser -D -s /bin/bash {username} && " f"mkdir -p /home/{username}/.ssh"
      )
    else:
      self.dockerfile_content.append(f"RUN useradd -m -s /bin/bash {username}")

    venv_path = f"/home/{username}/venv"
    self.dockerfile_content.extend(
      [
        # Create virtual environment
        f"ENV VIRTUAL_ENV={venv_path}",
        f"RUN python3 -m venv {venv_path}",
        # Make venv's Python the default python
        f"ENV PATH={venv_path}/bin:$PATH",
        "ENV PYTHONPATH=",  # Clear PYTHONPATH for clean venv
        # Ensure system uses venv's python/pip
        "RUN ln -sf /usr/bin/python3 /usr/local/bin/python",
        f"RUN ln -sf {venv_path}/bin/python3 /usr/local/bin/python3",
        f"RUN ln -sf {venv_path}/bin/pip /usr/local/bin/pip",
        f"RUN ln -sf {venv_path}/bin/pip3 /usr/local/bin/pip3",
      ]
    )

    # Add file operations
    home_path = f"/home/{username}"
    self.dockerfile_content.extend(
      [
        f"ADD --chown={username}:{username} documents.tar.gz {home_path}/",
        f"ADD --chown={username}:{username} tmp.tar.gz {home_path}/",
        f"ADD --chown={username}:{username} var.tar.gz {home_path}/",
        f"COPY --chown=root:root randomly_encrypt.sh {home_path}/",
        f"COPY --chown=root:root set_random_permissions.sh {home_path}/",
        f"COPY --chown={username}:{username} .ssh {home_path}/.ssh/",
        f"COPY --chown={username}:{username} requirements.txt {home_path}/",
        f"RUN chmod +x {home_path}/randomly_encrypt.sh",
        f"RUN chmod +x {home_path}/set_random_permissions.sh",
        f"RUN cd {home_path} && ./randomly_encrypt.sh",
        f"RUN cd {home_path} && ./set_random_permissions.sh",
        f"RUN chmod 700 {home_path}/.ssh",
      ]
    )

    # Install requirements in virtual environment
    self.dockerfile_content.append(
      f"RUN . {venv_path}/bin/activate && "
      f"pip install --no-cache-dir -r {home_path}/requirements.txt"
    )

    # Set working directory and default user
    self.dockerfile_content.extend(
      [
        f"WORKDIR {home_path}",
        f"USER {username}",
      ]
    )

    # Use bash as the entrypoint with virtual environment activation
    self.dockerfile_content.extend(
      [
        f'RUN echo "source {venv_path}/bin/activate" >> ~/.bashrc',
        'ENTRYPOINT ["bash"]',
        'CMD ["-l"]',  # Login shell to ensure .bashrc is sourced
      ]
    )

  def write_dockerfile(self) -> Path:
    """Write the Dockerfile to the build directory."""
    dockerfile_path = self.dst_dir / "Dockerfile"
    dockerfile_path.write_text("\n".join(self.dockerfile_content))

    logger.info(f"Dockerfile written to {dockerfile_path}")

    return dockerfile_path

  def build_and_run(
    self, image_name: str, container_name: str, dockerfile_hash: str
  ) -> None:
    """Build and run the Docker container."""
    final_image_name = f"{image_name}-hash{dockerfile_hash}"
    try:
      logger.info(f"Checking if image {final_image_name} exists")
      if check_image_exists(final_image_name):
        logger.info(f"Image {final_image_name} already exists")
      else:
        logger.info(f"Image {final_image_name} doesnt exist")
        build_cmd = ["docker", "build", "-t", final_image_name, "."]
        subprocess.run(build_cmd, check=True)

      run_cmd = [
        "docker",
        "run",
        "-it",  # Add interactive tty
        "-d",  # detached mode
        "--name",
        container_name,
        "--hostname",
        container_name,
        "-v",
        "/bin/df:/usr/bin/df:ro",
        "-v",
        "/bin/du:/usr/bin/du:ro",
        "-v",
        "/bin/free:/usr/bin/free:ro",
        final_image_name,
      ]

      subprocess.run(run_cmd, check=True)
      logger.info(f"Container {container_name} started")

    except subprocess.CalledProcessError as e:
      print(f"Error during docker operation: {e}")
      raise


def prepare_build_context(base_dir: Path | str, build_dir: Path | str) -> None:
  """Prepare the build context by copying required files."""
  base_dir = Path(base_dir)
  build_dir = Path(build_dir)

  if not build_dir.exists():
    build_dir.mkdir(parents=True)

    logger.info(f"Creating directory {build_dir}")

  required_files = [
    "documents.tar.gz",
    "tmp.tar.gz",
    "var.tar.gz",
    "randomly_encrypt.sh",
    "set_random_permissions.sh",
    ".ssh",
    "requirements.txt",
  ]

  for file in required_files:
    src_file = base_dir / file
    dst_file = build_dir / file

    logger.info(f"Copying {src_file} to {dst_file}")

    if src_file.exists():
      if src_file.is_dir():
        shutil.copytree(src_file, dst_file, dirs_exist_ok=True)
      else:
        shutil.copy2(src_file, dst_file)
    else:
      raise FileNotFoundError(f"Required file not found: {src_file}")


def build_and_run(image_name: str, container_name: str) -> None:
  """Build and run the Docker container using command line."""
  try:
    # Build the image
    build_cmd = ["docker", "build", "-t", image_name, "."]
    subprocess.run(build_cmd, check=True)

    # Run the container
    run_cmd = [
      "docker",
      "run",
      "-d",  # detached mode
      "--name",
      container_name,
      "--hostname",
      container_name,
      "-t",  # TTY
      "-v",
      "/bin/df:/usr/bin/df:ro",
      "-v",
      "/bin/du:/usr/bin/du:ro",
      "-v",
      "/bin/free:/usr/bin/free:ro",
      image_name,
    ]
    subprocess.run(run_cmd, check=True)

  except subprocess.CalledProcessError as e:
    print(f"Error during docker operation: {e}")
    raise


def run_python_container_test(
  container_name: str, user: str, test_file_name: str
) -> Tuple[float, str]:
  # Copy test file to container
  subprocess.run(
    [
      "docker",
      "cp",
      test_file_name,
      f"{container_name}:/home/{user}/test-container.py",
    ],
    check=True,
  )

  # Run test
  result = subprocess.run(
    ["docker", "exec", container_name, "python3", f"/home/{user}/test-container.py"],
    check=True,
    capture_output=True,
    text=True,
  )

  score = 0
  non_workings = ""
  # Parse and return results
  try:
    for line in result.stdout.strip().splitlines():
      if line.startswith("score:"):
        score = float(line.split(":")[1].strip())
      elif line.startswith("non_working_packages_names:"):
        non_workings = line.split(":")[1].strip()
  except Exception as e:
    logger.info(f"Error parsing score: {e}")
    logger.info(f"Full output:\n{result.stdout}")
    raise

  return score, non_workings


def run_shell_container_test(
  container_name: str, user: str, test_file_name: str
) -> None:
  pass


def kill_containers_with_the_name(name: str) -> None:
  print(f"Killing container with name: {name}")
  os.system(f"docker rm -f {name}")
  print(f"Container with name: {name} killed")


def check_image_exists(image_name: str) -> bool:
  """
  Check if a Docker image exists locally.

  Args:
      image_name: Name of the Docker image
      dockerfile_hash: Hash of the Dockerfile

  Returns:
      bool: True if image exists, False otherwise
  """
  client = docker.from_env()

  try:
    client.images.get(image_name)
    return True
  except docker.errors.ImageNotFound:
    return False


def hash_dockerfile(dockerfile_path: Path | str) -> str:
  """Hash the Dockerfile for 10 characters."""
  dockerfile_path = Path(dockerfile_path)

  return hashlib.sha256(dockerfile_path.read_text().encode()).hexdigest()[:10]


def main():
  user = "alice"
  container_name = "learn-container"

  failures = []
  successes = []

  filtered_images = [image for image in DOCKER_IMAGES if image["name"] not in skips]
  image_data = random.choice(filtered_images)
  # filtered_images = [image for image in DOCKER_IMAGES if image["name"] == image_name]
  # image_data = random.choice(filtered_images)

  logger.info(f"Picked image: {image_data['name']}")

  kill_containers_with_the_name("learn-container")

  try:
    base_image = image_data["name"]
    base_os = image_data["base_os"]

    source_dir = Path("./docker/fedora-learn-compose")
    dst_dir = Path("./docker/learn-compose")

    # Initialize handler and prepare everything
    docker_handler = DockerHandler(
      source_dir=source_dir,
      dst_dir=dst_dir,
    )

    # Prepare build environment
    docker_handler.prepare_build_context()

    # Generate and write Dockerfile
    docker_handler.generate_dockerfile(base_image, base_os, user)
    dockerfile_path = docker_handler.write_dockerfile()
    dockerfile_hash = hash_dockerfile(dockerfile_path)

    logger.info(f"Generated Dockerfile for {base_image}")

    # Build and run
    os.chdir(docker_handler.dst_dir)
    docker_handler.build_and_run(base_image, container_name, dockerfile_hash)
    logger.info(f"Container {base_image} started")

    score, non_workings = run_python_container_test(
      container_name, user, "test-container.py"
    )

    logger.info(f"Score: {score}")
    logger.info(f"Non working packages: {non_workings}")

    successes.append(base_image)
  except Exception as e:
    print(f"Error during docker operation: {e}")
    failures.append(base_image)


if __name__ == "__main__":
  main()
