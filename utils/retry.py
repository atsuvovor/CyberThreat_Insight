#utils/retry.py
#Error Recovery + Retry Logic

import time
from utils.logger import setup_logger

logger = setup_logger("retry")

def retry(max_attempts=3, delay=5):
    """Retry decorator for transient failures"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {e}")
                    if attempt < max_attempts:
                        logger.info(f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error("Max attempts reached. Raising exception.")
                        raise
        return wrapper
    return decorator
