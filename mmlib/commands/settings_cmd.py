"""set and config - reading and writing the cascade."""

from __future__ import annotations

import shlex

from ..config import save_config
from ..console import die
from ..launch import tool_for
from ..resolve import resolve
from ..runtimes import load_runtimes
from ..settings import (flags_for, parse_value, pick_runtime, resolve_settings,
                        runner_name, scopes, vocab)


def settings_layer(scope: str, entries, cfg, runtimes, create: bool) -> dict:
    """Return the dict backing one layer of the cascade."""
    s = cfg["settings"]
    if scope == "@defaults":
        return s["defaults"]
    if scope.startswith("@"):
        runner = scope[1:]
        return s["runner"].setdefault(runner, {}) if create else s["runner"].get(runner, {})
    handle = resolve(scope, entries, cfg).handle
    return s["model"].setdefault(handle, {}) if create else s["model"].get(handle, {})


def cmd_set(args, entries, cfg):
    runtimes = load_runtimes(cfg)
    known = scopes(runtimes)
    if args.scope.startswith("@") and args.scope not in known:
        die(f"unknown scope {args.scope!r} (use {', '.join(known)} or a model handle)")
    layer = settings_layer(args.scope, entries, cfg, runtimes, create=True)

    for key in args.clear:
        layer.pop(key, None)

    if args.raw is not None:
        if args.raw:
            layer["raw"] = shlex.split(args.raw)
        else:
            layer.pop("raw", None)

    allowed = vocab(runtimes)
    for pair in args.assignments:
        if "=" not in pair:
            die(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        if key not in allowed:
            die(f"unknown setting {key!r}\n     known: {', '.join(allowed)}"
                f"\n     anything else goes through --raw")
        if key == "runtime" and value not in runtimes:
            die(f"unknown runtime {value!r} (known: {', '.join(sorted(runtimes))})")
        layer[key] = value if key == "runtime" else parse_value(value)

    # Don't leave empty layers cluttering the file.
    if not layer:
        for group in ("runner", "model"):
            cfg["settings"][group] = {k: v for k, v in cfg["settings"][group].items() if v}
    save_config(cfg)

    label = (args.scope if args.scope.startswith("@")
             else resolve(args.scope, entries, cfg).handle)
    if not layer:
        print(f"{label}: empty")
        return
    for key in sorted(layer):
        value = " ".join(layer[key]) if key == "raw" else layer[key]
        print(f"{label}: {key} = {value}")
    if not args.scope.startswith("@"):
        print(f"(mm config {label} shows the merged result)")


def cmd_config(args, entries, cfg):
    """Show the settings a model will actually run with, and where each came from."""
    e = resolve(args.handle, entries, cfg)
    runtimes = load_runtimes(cfg)
    runtime = pick_runtime(e, cfg, runtimes)
    serve = getattr(args, "serve", False)
    tool = tool_for(e, serve, args, runtime)
    resolved, raw = resolve_settings(e, cfg, runner_name(e, runtime))
    specs = flags_for(tool, runtime)

    where = f"{runtime.name}:{tool}" if runtime else tool
    print(f"{e.handle}  ->  {where}")
    if runtime:
        print(f"  {'venv':<12}{runtime.venv}")
    if not resolved and not raw:
        print("  (no settings; runner defaults apply)")
        return

    rows = []
    for key in sorted(resolved):
        value, source = resolved[key]
        if key == "runtime":
            note = ""
        else:
            note = "" if key in specs else f"ignored - {where} has no {key}"
        rows.append((key, str(value), source, note))
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    for key, value, source, note in rows:
        line = (f"  {key.ljust(widths[0])}  {value.ljust(widths[1])}  "
                f"{source.ljust(widths[2])}")
        if note:
            line += f"  {note}"
        print(line.rstrip())
    if raw:
        print(f"  {'raw'.ljust(widths[0])}  {' '.join(raw)}")
