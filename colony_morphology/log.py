from time import perf_counter
from contextlib import contextmanager
from typing import Dict, Optional

@contextmanager
def timer(label: str, verbose: bool = False, timings: Optional[Dict[str, float]] = None):
    """
    Context manager to measure elapsed time for a block of code.

    Args:
        label: Name of the block being measured.
        verbose: If True, prints timing immediately.
        timings: Optional dict to accumulate timings by label.
    """
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        if timings is not None:
            timings[label] = dt
        if verbose:
            print(f"{label}: {dt:.3f}s", flush=True)
