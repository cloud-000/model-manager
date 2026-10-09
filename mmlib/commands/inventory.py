"""ls, path, link - the commands that only look at what is on disk."""

from __future__ import annotations

import os

from ..console import die
from ..paths import HUB, LINK_DIR, expand
from ..render import fmt_age, fmt_ctx, human, table
from ..resolve import resolve
from ..runtimes import load_runtimes
from ..scan import disk_usage
from ..settings import pick_runtime


def cmd_ls(args, entries, cfg):
    if args.names:
        for e in entries:
            if e.runnable or args.all:
                print(e.handle)
        return

    runtimes = load_runtimes(cfg)

    shown = [e for e in entries if e.role != "none"]
    if not args.all:
        shown = [e for e in shown if e.role != "draft"]
    order = {"mlx": 0, "hf": 1, "gguf": 2, "litert": 3, "ct2": 4}
    shown.sort(key=lambda e: (order.get(e.fmt, 9), -e.size))

    # The runtime column only earns its width once there is a choice to make.
    multi = len(runtimes) > 1

    def runtime_of(e):
        # Only the MLX runner has runtimes to choose between; the rest are
        # singletons and their column is empty.
        rt = pick_runtime(e, cfg, runtimes)
        return rt.name if rt else "-"

    # The format lives in the handle now, so this column reports the role.
    head = ("HANDLE", "ROLE", "QUANT", "SIZE", "CTX", "LAST", "NOTE")
    rows = []
    for e in shown:
        row = [e.handle, e.role, e.quant or "-", human(e.size), fmt_ctx(e.ctx),
               fmt_age(cfg["last_used"].get(e.handle)), cfg["notes"].get(e.handle, "")]
        if multi:
            row.insert(2, runtime_of(e))
        rows.append(tuple(row))
    if multi:
        head = head[:2] + ("RUNTIME",) + head[2:]

    line = table(rows, head)

    skipped = [e for e in entries if e.role == "none"]
    if skipped:
        print(f"\nnot runnable here ({len(skipped)}):")
        for e in sorted(skipped, key=lambda e: -e.size):
            cells = [f"  {e.fmt}/{e.slug}", "", "", human(e.size), "", "", e.reason or ""]
            if multi:
                cells.insert(2, "")
            line(cells, dim=True)

    usage = disk_usage(entries)
    total = sum(n for _, n, _ in usage)
    print(f"\n{len(shown)} runnable, {human(total)} on disk"
          + (f" in {HUB}" if len(usage) <= 1 else ""))
    if len(usage) > 1:
        for kind, n, where in usage:
            print(f"  {kind:<6} {human(n):>7}  {where}")


def cmd_path(args, entries, cfg):
    print(resolve(args.handle, entries, cfg).path)


def cmd_link(args, entries, cfg):
    LINK_DIR.mkdir(parents=True, exist_ok=True)

    # Drop only the links we own: symlinks pointing into a source mm manages.
    # One level deep only, and never through a symlink, so this can't wander
    # into the cache itself.
    owned = [str(HUB)] + [str(expand(r)) for r in cfg.get("roots", [])]
    dirs = [LINK_DIR] + [d for d in LINK_DIR.iterdir()
                         if d.is_dir() and not d.is_symlink()]
    for parent in dirs:
        for p in parent.iterdir():
            if p.is_symlink() and any(o in os.readlink(p) for o in owned):
                p.unlink()
    for d in dirs[1:]:
        if not any(d.iterdir()):
            d.rmdir()

    made = 0
    for e in entries:
        if not e.runnable:
            continue
        fmt_dir = LINK_DIR / e.fmt          # ~/models/mlx, ~/models/gguf, ...
        fmt_dir.mkdir(exist_ok=True)
        name = e.slug + (f"-{e.variant}" if e.variant else "")
        if e.fmt == "gguf":
            name += ".gguf"
        target = fmt_dir / name
        if target.exists() or target.is_symlink():
            die(f"{target} exists and is not a link mm manages")
        target.symlink_to(e.path)
        made += 1
    print(f"linked {made} models into {LINK_DIR}")
