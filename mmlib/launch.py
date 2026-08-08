"""Building the command line for one model and handing the process over.

The runner-specific parts live in runners/; this file is the part that is the
same whatever launches the model - resolve, translate the cascade, warn about
what got dropped, let `--` win, exec.
"""

from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime, timezone

from . import runners
from .config import save_config
from .console import die, warn
from .resolve import resolve
from .runtimes import Runtime, load_runtimes
from .settings import (flags_for, pick_runtime, resolve_settings, runner_name,
                       settings_flags)


def tool_for(e, serve: bool, args, runtime: Runtime | None) -> str:
    runner = runners.for_format(e.fmt)
    return runner.tool_for(e, serve, args) if runner else "?"


def build_command(e, serve: bool, args, runtime: Runtime | None,
                  runtimes: dict) -> tuple[list[str], str]:
    reason = runners.check(e, runtimes)
    if reason:
        die(f"{e.handle} cannot run here: {reason}")
    runner = runners.for_format(e.fmt)
    return runner.command(e, serve, args, runtime), runner.tool_for(e, serve, args)


def launch(args, entries, cfg, serve: bool):
    e = resolve(args.handle, entries, cfg)
    runtimes = load_runtimes(cfg)
    runtime = pick_runtime(e, cfg, runtimes)
    cmd, tool = build_command(e, serve, args, runtime, runtimes)

    resolved, raw = resolve_settings(e, cfg, runner_name(e, runtime))
    groups, dropped = settings_flags(resolved, tool, runtime)

    # Only suggest serving if this model's *own* server tool would accept the key.
    server_tool = tool_for(e, True, args, runtime)
    for key in dropped:
        where = f"{tool} (runtime {runtime.name})" if runtime else tool
        hint = ("" if serve or key not in flags_for(server_tool, runtime)
                else " (try mm serve)")
        warn(f"{key} is not supported by {where}, ignored{hint}")

    # Explicit passthrough beats the cascade. Both runners honour last-wins, but
    # llama.cpp grumbles about repeated flags, so drop ours where they collide.
    flags = [x for g in groups if g[0] not in args.extra for x in g]
    cmd += flags + raw + list(args.extra)

    print("$ " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    if args.dry_run:
        return

    cfg["last_used"][e.handle] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_config(cfg)
    os.execvp(cmd[0], cmd)
