from datetime import datetime
import subprocess
import time
from loguru import logger
import schedule
import psutil
import os
import signal
from threading import Timer

os.environ["TZ"] = "Asia/Jakarta"  # or 'Asia/Singapore'

logger.add("logs/infinite_loop.log")

TIMEOUT_MINUTES = 45


def check_if_we_have_enough_space():
  disk_usage = psutil.disk_usage("/")
  free_space_gb = disk_usage.free / (1024**3)
  return free_space_gb > 1


def kill_process(process):
  logger.warning(f"Job exceeded {TIMEOUT_MINUTES} minutes, terminating...")
  process.terminate()
  try:
    process.wait(timeout=5)  # Give it 5 seconds to terminate gracefully
  except subprocess.TimeoutExpired:
    logger.warning("Process didn't terminate gracefully, killing...")
    process.kill()  # Force kill if it doesn't terminate


def job():
  if not check_if_we_have_enough_space():
    logger.info("Not enough space, skipping")
    return

  try:
    # Start the process
    process = subprocess.Popen(
      ["./.venv/bin/python3", "scripts/main.py"],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )

    # Set up the timeout
    timer = Timer(TIMEOUT_MINUTES * 60, kill_process, [process])
    timer.start()

    try:
      stdout, stderr = process.communicate()
      timer.cancel()  # Cancel the timer if process completes normally

      if process.returncode == 0:
        logger.info(
          f"Process completed successfully. Log length: {len(stdout.splitlines())}"
        )
        if stdout.splitlines():
          logger.info(f"Last log: {stdout.splitlines()[-1].splitlines()[-1]}")
      else:
        logger.error(
          f"Process failed with code {process.returncode}. " f"Error: {stderr}"
        )

    except subprocess.TimeoutExpired:
      logger.error(f"Process timed out after {TIMEOUT_MINUTES} minutes")
      process.kill()
      stdout, stderr = process.communicate()

  except Exception as e:
    logger.error(f"Exception: {e}")

  logger.info(f"Job completed at {datetime.now()}")


if __name__ == "__main__":
  wait_time = int(60 * 5)

  logger.info("Running the first job immediately")

  # Run the job immediately
  cur_time = datetime.now()
  job()
  logger.info(f"Time taken: {datetime.now() - cur_time}")

  while True:
    logger.info(f"Waiting for {wait_time} seconds")
    time.sleep(wait_time)

    cur_time = datetime.now()
    job()
    logger.info(f"Time taken: {datetime.now() - cur_time}")
