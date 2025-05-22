import subprocess
import plotext as plt
import time

WINDOW_SIZE = 20
CONTAINER_NAME = "space-warmer-container"
DIR = "/usr/src/app"


def get_directory_size(container_name, directory):
  # Execute the du command inside the specified Docker container
  cmd = ["docker", "exec", container_name, "du", "-sb", directory]  # '-s' for summary
  result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
  # Parse the output to extract the total size in kilobytes
  if result.stdout:
    size = int(result.stdout.split()[-2])  # Get the first token which is the size
    return size


def plot_directory_size(container_name, directory):
  times = []
  sizes = []
  start_time = time.time()

  try:
    while True:
      # Get directory size
      new_size = get_directory_size(container_name, directory)
      new_time = time.time() - start_time

      sizes.append(new_size)
      times.append(new_time)

      if len(times) > WINDOW_SIZE:
        sizes.pop(0)
        times.pop(0)

      # Clear the plot, plot the new data
      plt.clear_figure()

      plt.title("Real-Time Disk Plot")
      plt.xlabel("Time (s)")
      plt.ylabel(f"Storage size of {DIR} (bytes)")

      # Plot the data
      plt.plot(times, sizes)

      # Display the plot
      plt.show()

      # Pause for a short duration to simulate real-time update
      time.sleep(0.01)

  except KeyboardInterrupt:
    print("Stopped plotting directory size.")


if __name__ == "__main__":
  # container_name = input("Enter the Docker container name or ID: ")
  # directory = input("Enter the directory path inside the container: ")
  plot_directory_size(CONTAINER_NAME, DIR)
