from contextlib import contextmanager
from datetime import datetime
import random
import string
from typing import Callable


def nanoid(size=21) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


@contextmanager
def timeout(seconds: int, callback: Callable = lambda: print()):
    """
    Context manager that raises a TimeoutError if the code inside the context takes longer than the specified time.

    This implementation uses threading.Timer which is thread-safe, unlike signal-based approaches.

    Args:
        seconds (int): Maximum number of seconds to allow the code to run

    Yields:
        None: The context to execute code within the timeout constraint

    Raises:
        TimeoutError: If the code execution exceeds the specified timeout

    Example:
        >>> with timeout(5):
        ...     # Code that should complete within 5 seconds
        ...     long_running_function()
    """
    import threading

    timer = None
    exception = TimeoutError(f"Execution timed out after {seconds} seconds")

    def timeout_handler():
        nonlocal timer
        callback()
        raise exception

    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
    finally:
        if timer:
            timer.cancel()


def int_to_ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def unflatten_toml_dict(d: dict) -> dict:
    result = {}
    for key, value in d.items():
        parts = key.split(".")
        current_level = result
        for i, part in enumerate(parts):
            if i == len(parts) - 1:  # Last part, assign value
                current_level[part] = value
            else:
                current_level = current_level.setdefault(part, {})

    return result
