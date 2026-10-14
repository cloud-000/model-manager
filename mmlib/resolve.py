"""Turning what the user typed into exactly one Entry."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .console import die
from .entry import Entry


def resolve(query: str, entries: list[Entry], cfg: dict) -> Entry:
    query = cfg.get("aliases", {}).get(query, query)
    q = query.lower()

    for e in entries:                                    # exact handle
        if e.handle.lower() == q:
            return e
    for e in entries:                                    # exact repo id
        if e.repo.lower() == q:
            return e

    # 1. Direct filesystem path (directory or weights file)
    p = Path(os.path.expanduser(query)).resolve()
    if p.exists():
        if p.is_dir():
            from .classify import classify_dir, classify_weights_dir, inspect_dir
            from .scan import make_slug

            info = inspect_dir(p)
            entry = None
            if info.has_config:
                entry = classify_dir(p, repo=f"local/{p.name}", variant=None, info=info)
            if entry is None and (info.ggufs or info.litert):
                cand = classify_weights_dir(p, repo=f"local/{p.name}", variant=None, info=info)
                if cand:
                    entry = cand[0]
            if entry:
                entry.source = "local"
                entry.slug = make_slug(p.name)
                return entry
            die(f"{p} is a directory but contains no recognized model weights or config.json")
        elif p.is_file():
            from .scan import make_slug

            if p.name.endswith(".gguf"):
                return Entry(f"local/{p.stem}", "gguf", "chat", p,
                             quant=None, size=p.stat().st_size, slug=make_slug(p.stem),
                             source="local")
            elif p.name.endswith(".litertlm"):
                return Entry(f"local/{p.stem}", "litert", "chat", p,
                             quant=None, size=p.stat().st_size, slug=make_slug(p.stem),
                             source="local")
            die(f"{p} is not a recognized model file (expected .gguf or .litertlm)")

    hits = [e for e in entries if q in e.handle.lower() or q in e.repo.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        runnable = [e for e in hits if e.runnable]
        if len(runnable) == 1:
            return runnable[0]
        print(f"mm: {query!r} is ambiguous:", file=sys.stderr)
        for e in hits:
            print(f"      {e.handle}", file=sys.stderr)
        sys.exit(1)

    # 2. Remote Hugging Face repo ID or handle (on-demand entry)
    raw_repo = query[3:] if query.startswith("hf/") else query
    parts = raw_repo.split("/")
    if len(parts) == 2 and parts[0] and parts[1]:
        from .scan import make_slug

        return Entry(repo=raw_repo, fmt="hf", role="chat", path=Path(raw_repo),
                     slug=make_slug(parts[1]), source="hub")

    die(f"no model matching {query!r}  (try: mm ls)")
