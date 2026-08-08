"""Turning what the user typed into exactly one Entry."""

from __future__ import annotations

import sys

from .console import die
from .entry import Entry


def resolve(query: str, entries: list[Entry], cfg: dict) -> Entry:
    query = cfg["aliases"].get(query, query)
    q = query.lower()

    for e in entries:                                    # exact handle
        if e.handle.lower() == q:
            return e
    for e in entries:                                    # exact repo id
        if e.repo.lower() == q:
            return e

    hits = [e for e in entries if q in e.handle.lower() or q in e.repo.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        die(f"no model matching {query!r}  (try: mm ls)")

    # A bare slug that names exactly one runnable variant is unambiguous enough.
    runnable = [e for e in hits if e.runnable]
    if len(runnable) == 1:
        return runnable[0]

    print(f"mm: {query!r} is ambiguous:", file=sys.stderr)
    for e in hits:
        print(f"      {e.handle}", file=sys.stderr)
    sys.exit(1)
