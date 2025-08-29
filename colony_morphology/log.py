from time import perf_counter
from contextlib import contextmanager
from typing import Dict, Optional
import sys

@contextmanager
def timer(
    label: str,
    verbose: bool = False,
    timings: Optional[Dict[str, float]] = None,
    *,
    inline: bool = True,           # True = print on one line: "label ... 0.123s"
    stream = sys.stdout
):
    t0 = perf_counter()
    if verbose:
        if inline:
            print(f"{label} ... ", end="", file=stream, flush=True)
        else:
            print(f"[START] {label}", file=stream, flush=True)
    try:
        yield
    except BaseException:
        dt = perf_counter() - t0
        if timings is not None:
            timings[label] = dt
        if verbose:
            if inline:
                print(f"FAILED after {dt:.3f}s", file=stream, flush=True)
            else:
                print(f"[FAIL ] {label}: {dt:.3f}s", file=stream, flush=True)
        raise
    else:
        dt = perf_counter() - t0
        if timings is not None:
            timings[label] = dt
        if verbose:
            if inline:
                print(f"{dt:.3f}s", file=stream, flush=True)
            else:
                print(f"[DONE ] {label}: {dt:.3f}s", file=stream, flush=True)
