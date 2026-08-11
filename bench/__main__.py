"""Run every benchmark: ``python -m bench``."""
from __future__ import annotations

import os
import sys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    top = os.path.dirname(here)
    if top not in sys.path:
        sys.path.insert(0, top)

    from bench import host_dispatch
    from bench._harness import render

    print("volkano host-side microbenchmarks (ns per call, median of reps)")
    sections = host_dispatch.collect()
    render(sections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
