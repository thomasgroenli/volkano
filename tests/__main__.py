"""Run every test in this package: ``python -m tests``."""
from __future__ import annotations

import os
import sys
import unittest


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    top = os.path.dirname(here)
    if top not in sys.path:
        sys.path.insert(0, top)
    suite = unittest.TestLoader().discover(start_dir=here, top_level_dir=top)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
