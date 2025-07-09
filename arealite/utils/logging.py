import time
from contextlib import contextmanager


@contextmanager
def record_timing(name, timing_stats):
    start_time = time.perf_counter()
    yield
    timing_stats[name] = time.perf_counter() - start_time
