import docker
import time
from datetime import datetime
import plotext as plt

WINDOW_SIZE = 20


def get_container_stats(container):
  """Get formatted container stats"""
  stats = container.stats(stream=False)

  # Memory stats in MB
  mem_usage = stats["memory_stats"].get("usage", 0) / (1024 * 1024)
  mem_limit = stats["memory_stats"].get("limit", 0) / (1024 * 1024)
  mem_percent = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0

  # CPU stats
  cpu_delta = (
    stats["cpu_stats"]["cpu_usage"]["total_usage"]
    - stats["precpu_stats"]["cpu_usage"]["total_usage"]
  )
  system_delta = (
    stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
  )
  cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0

  # Get storage info
  df_output = container.exec_run("df -m /").output.decode()
  used_storage = 0
  try:
    storage_line = df_output.strip().split("\n")[1]
    used_storage = int(storage_line.split()[2])  # Used MB
  except:
    pass

  return {
    "memory_mb": round(mem_usage, 2),
    "cpu_percent": round(cpu_percent, 2),
    "storage_mb": used_storage,
  }


def plot_container_metric(container_name, metric_type="cpu"):
  """
  Plot a specific container metric.
  metric_type can be: 'cpu', 'memory', or 'storage'
  """
  client = docker.from_env()
  container = client.containers.get(container_name)

  times = []
  values = []
  start_time = time.time()

  # Set up metric-specific details
  if metric_type == "cpu":
    ylabel = "CPU Usage (%)"
    key = "cpu_percent"
  elif metric_type == "memory":
    ylabel = "Memory Usage (MB)"
    key = "memory_mb"
  else:  # storage
    ylabel = "Storage Used (MB)"
    key = "storage_mb"

  try:
    while True:
      # Get new stats
      stats = get_container_stats(container)
      current_time = time.time() - start_time

      # Append new values
      times.append(current_time)
      values.append(stats[key])

      # Maintain window size
      if len(times) > WINDOW_SIZE:
        times.pop(0)
        values.pop(0)

      # Clear and create new plot
      plt.clear_figure()
      plt.clf()

      plt.plot(times, values)
      plt.title(f"Container {metric_type.upper()} Stats: {container_name}")
      plt.xlabel("Time (s)")
      plt.ylabel(ylabel)

      plt.show()
      time.sleep(1)

  except KeyboardInterrupt:
    print("\nStopped monitoring.")
  finally:
    client.close()


def main():
  client = docker.from_env()

  # Create the test container
  storage_test_script = """
while true; do
    # Create a file with random size (1-100MB)
    size=$((RANDOM % 100 + 1))
    dd if=/dev/urandom of=/tmp/test$RANDOM bs=1M count=$size 2>/dev/null
    
    # CPU work
    dd if=/dev/zero of=/dev/null bs=1M count=100 2>/dev/null
    
    sleep 1
    
    # Random deletions
    find /tmp -type f -name 'test*' -print0 | shuf -z -n $((RANDOM % 5)) | xargs -0 rm -f 2>/dev/null
done
    """

  container = client.containers.run(
    "alpine:latest",
    command=f'sh -c "{storage_test_script}"',
    detach=True,
    remove=True,
    name="stats-test",
    mem_limit="512m",
    cpu_period=100000,
    cpu_quota=50000,
  )

  print("\nWhat would you like to monitor?")
  print("1. CPU usage")
  print("2. Memory usage")
  print("3. Storage usage")

  choice = input("Enter your choice (1-3): ")

  metric_map = {"1": "cpu", "2": "memory", "3": "storage"}

  metric = metric_map.get(choice, "cpu")
  print(f"\nStarting {metric} monitoring. Press Ctrl+C to stop.")

  try:
    plot_container_metric("stats-test", metric)
  finally:
    try:
      container.stop()
      print("Container stopped and removed")
    except:
      pass


if __name__ == "__main__":
  main()
