"""``python -m volkano`` — the maintenance CLI.

One subcommand, ``update``: refetch the XML the package is configured
to use and rewrite ``__init__.pyi`` from it. Importing volkano does
neither, so this is the command that moves both forward.

The implementation lives in :mod:`volkano.stub` so it can be imported
and tested without spawning a subprocess; this module is only the
``-m`` entry point.
"""

from __future__ import annotations

import sys

from .stub import main


if __name__ == '__main__':
    sys.exit(main())
