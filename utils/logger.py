import logging
import time
from functools import wraps

def setup_logger(name="cyberthreat_insight", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def timeit(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)
            logger.info(f"{func.__name__} completed in {duration}s")
            return result
        return wrapper
    return decorator
