import cProfile
import time
from functools import wraps


def count_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} execution time elapsed {end_time-start_time}s")
        return result
    return wrapper
