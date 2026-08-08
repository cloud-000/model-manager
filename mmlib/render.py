"""Turning numbers and rows into something readable in a terminal."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def human(n: int) -> str:
    if n <= 0:
        return "-"
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit in ("B", "K") else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}G"


def fmt_ctx(n: int | None) -> str:
    """Context lengths are quoted both ways: 131072 and 128000 are both '128k'."""
    if not n:
        return "-"
    if n % 1000 == 0:
        return f"{n // 1000}k"
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def fmt_age(stamp: str | None) -> str:
    if not stamp:
        return "-"
    try:
        delta = datetime.now(timezone.utc) - datetime.fromisoformat(stamp)
    except ValueError:
        return "-"
    mins = int(delta.total_seconds() // 60)
    for size, unit in ((1440, "d"), (60, "h")):
        if mins >= size:
            return f"{mins // size}{unit}"
    return f"{max(mins, 0)}m"


def table(rows, head):
    """Print aligned columns, returning a printer for extra rows of the same shape."""
    widths = [max(len(str(r[i])) for r in list(rows) + [head]) for i in range(len(head))]

    def line(cells, dim=False):
        text = "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
        print(f"\033[2m{text}\033[0m" if dim and sys.stdout.isatty() else text)

    line(head)
    for r in rows:
        line(r)
    return line
