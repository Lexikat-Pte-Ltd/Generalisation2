import subprocess
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Configuration
CONTAINER_NAME = "space-warmer-container"
DIR = "/usr/src/app"
PUSH_GATEWAY = "localhost:9091"  # Update this with your Prometheus Push Gateway address
METRIC_PREFIX = "container"  # Prefix for metric names
PUSH_INTERVAL = 1  # Seconds between pushes


def get_directory_size(container_name, directory):
  """Get directory size from Docker container in bytes"""
  cmd = ["docker", "exec", container_name, "du", "-sb", directory]
  result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
  if result.stdout:
    size = int(result.stdout.split()[-2])
    return size
  return 0


def monitor_and_push_metrics():
  """Monitor directory size and push to Prometheus"""
  # Create a new registry
  registry = CollectorRegistry()

  # Create metrics
  dir_size = Gauge(
    f"{METRIC_PREFIX}_directory_size_bytes",
    "Directory size in bytes",
    ["container", "directory"],
    registry=registry,
  )

  # Create metadata labels
  labels = {"container": CONTAINER_NAME, "directory": DIR}

  print(f"Starting to monitor directory size for container: {CONTAINER_NAME}")
  print(f"Pushing metrics to: {PUSH_GATEWAY}")

  try:
    while True:
      try:
        # Get current size
        size = get_directory_size(CONTAINER_NAME, DIR)

        # Update metric
        dir_size.labels(container=CONTAINER_NAME, directory=DIR).set(size)

        # Push to gateway
        push_to_gateway(PUSH_GATEWAY, job="directory_size_monitor", registry=registry)

        print(f"Pushed metric: {size} bytes at {time.strftime('%Y-%m-%d %H:%M:%S')}")

      except Exception as e:
        print(f"Error pushing metrics: {e}")

      time.sleep(PUSH_INTERVAL)

  except KeyboardInterrupt:
    print("\nStopping metric collection...")

    # Clean up on exit (optional)
    try:
      # Delete metrics for this job when stopping
      # Note: Uncomment if you want metrics to be removed when the script stops
      # delete_from_gateway(PUSH_GATEWAY, job='directory_size_monitor')
      pass
    except:
      print("Could not clean up metrics from Push Gateway")


if __name__ == "__main__":
  monitor_and_push_metrics()
