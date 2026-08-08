"""Error reporting. Kept in its own module because everything imports it."""

from __future__ import annotations

import sys


def die(msg: str, code: int = 1):
    print(f"mm: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str):
    print(f"mm: warning: {msg}", file=sys.stderr)
