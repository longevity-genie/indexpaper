import time
from functools import wraps
from typing import Optional

from loguru import logger


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def timing(custom_message: Optional[str] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)  # Call the original function
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            message = f"{custom_message if custom_message is not None else ''} {func.__name__} took {format_time(execution_time)}"
            logger.debug(message)
            return result
        return wrapper
    return decorator