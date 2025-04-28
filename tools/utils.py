import time
from functools import wraps

# Rate limiting decorator
def rate_limited(calls_per_second):
    """
    Decorator that limits the rate at which a function can be called.

    Args:
        calls_per_second (float): The maximum number of calls allowed per second.
    """
    interval = 1.0 / calls_per_second

    def decorator(func):
        last_called = [0.0]  # Use a list to allow modification within the closure

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator