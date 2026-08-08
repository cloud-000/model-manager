"""note, alias, rm, root, runtime - the commands that edit config.json."""

from __future__ import annotations

import shutil

from ..classify import dir_size, repo_size
from ..config import save_config
from ..console import die
from ..paths import expand
from ..render import human
from ..resolve import resolve
from ..runtimes import load_runtimes


def cmd_note(args, entries, cfg):
    e = resolve(args.handle, entries, cfg)
    if args.clear:
        cfg["notes"].pop(e.handle, None)
    elif args.text:
        cfg["notes"][e.handle] = " ".join(args.text)
    else:
        print(cfg["notes"].get(e.handle, ""))
        return
    save_config(cfg)


def cmd_alias(args, entries, cfg):
    if args.rm:
        if cfg["aliases"].pop(args.rm, None) is None:
            die(f"no alias {args.rm!r}")
        save_config(cfg)
    elif args.name and args.handle:
        target = resolve(args.handle, entries, cfg)
        cfg["aliases"][args.name] = target.handle
        save_config(cfg)
        print(f"{args.name} -> {target.handle}")
    else:
        for name, target in sorted(cfg["aliases"].items()):
            print(f"{name:<16} {target}")


def cmd_rm(args, entries, cfg):
    e = resolve(args.handle, entries, cfg)
    target = e.container
    if target is None or not target.is_dir():
        die(f"cannot locate the directory holding {e.repo}")

    size = repo_size(target) if e.source == "hub" else dir_size(target)
    siblings = [x.handle for x in entries if x.container == target]
    print(f"delete {target}")
    print(f"  repo:    {e.repo}")
    print(f"  source:  {e.source}")
    print(f"  size:    {human(size)}")
    print(f"  removes: {', '.join(siblings)}")
    if e.source == "local":
        print("  note:    this is a directory you registered, not a cache entry")
    if not args.yes:
        if input("type the repo name to confirm: ").strip() != e.repo:
            die("aborted")
    shutil.rmtree(target)
    for handle in siblings:
        cfg["notes"].pop(handle, None)
        cfg["last_used"].pop(handle, None)
    save_config(cfg)
    print(f"removed {e.repo}")


# --------------------------------------------------------------------------
# roots
# --------------------------------------------------------------------------

def cmd_root(args, entries, cfg):
    roots = cfg["roots"]
    if args.action == "add":
        path = expand(args.path)
        if not path.is_dir():
            die(f"not a directory: {path}")
        if str(path) in roots:
            die(f"already registered: {path}")
        roots.append(str(path))
        save_config(cfg)
        print(f"added {path}  (re-run `mm ls` to see what it holds)")
    elif args.action == "rm":
        path = str(expand(args.path))
        if path not in roots:
            die(f"not registered: {path}")
        roots.remove(path)
        save_config(cfg)
        print(f"removed {path}")
    else:
        if not roots:
            print("no roots registered; mm is reading the hub cache only")
        for r in roots:
            found = sum(1 for e in entries if e.source == "local"
                        and str(e.container).startswith(r))
            print(f"{r}  ({found} model{'' if found == 1 else 's'})")


# --------------------------------------------------------------------------
# runtimes
# --------------------------------------------------------------------------

def runtime_flag(args, cfg, runtimes):
    """Teach one runtime a setting the base vocabulary does not have.

    A fork's extra flags belong to the fork, not to mm: declaring them here
    keeps `mm set` able to reject a typo, without mm carrying a table of every
    downstream mlx-lm in existence.
    """
    if args.name not in runtimes:
        die(f"no runtime {args.name!r} (add it first: mm runtime add {args.name} <venv>)")
    flags = cfg["runtimes"].setdefault(args.name, {}).setdefault("flags", {})

    if args.rm:
        if flags.pop(args.rm, None) is None:
            die(f"runtime {args.name} has no flag {args.rm!r}")
        save_config(cfg)
        print(f"{args.name}: dropped {args.rm}")
        return

    if not args.spec:
        for key, spec in sorted(flags.items()):
            print(f"{args.name}: {key} = {spec}")
        return

    if "=" not in args.spec:
        die(f"expected key=--flag, got {args.spec!r}")
    key, _, flag = args.spec.partition("=")
    if not flag.startswith("-"):
        die(f"{flag!r} does not look like a flag")
    spec = {"flag": flag, "type": "bool" if args.bool else "value"}
    if args.negatable:
        if not args.bool:
            die("--negatable only makes sense with --bool")
        spec["negatable"] = True
    if args.tools:
        spec["tools"] = [t.strip() for t in args.tools.split(",") if t.strip()]
    flags[key] = spec
    save_config(cfg)
    print(f"{args.name}: {key} -> {flag} ({spec['type']})")
    print(f"  use it with: mm set <handle> {key}="
          + ("true" if args.bool else "<value>"))


def cmd_runtime(args, entries, cfg):
    runtimes = load_runtimes(cfg)

    if args.action == "add":
        venv = expand(args.venv)
        if not (venv / "bin").is_dir():
            die(f"not a venv (no bin/): {venv}")
        cfg["runtimes"].setdefault(args.name, {})["venv"] = str(venv)
        save_config(cfg)
        rt = load_runtimes(cfg)[args.name]
        extra = sorted(rt.architectures() - runtimes["mlx"].architectures())
        print(f"added runtime {args.name} -> {venv}")
        if extra:
            print(f"  adds architectures: {', '.join(extra)}")
        print(f"  pin a model to it with: mm set <handle> runtime={args.name}")
        return

    if args.action == "rm":
        if args.name == "mlx":
            die("the default `mlx` runtime cannot be removed (set MM_MLX_VENV instead)")
        if cfg["runtimes"].pop(args.name, None) is None:
            die(f"no runtime {args.name!r}")
        save_config(cfg)
        print(f"removed runtime {args.name}")
        return

    if args.action == "flag":
        return runtime_flag(args, cfg, runtimes)

    base = runtimes["mlx"].architectures()
    for name, rt in sorted(runtimes.items()):
        models = rt.models_dir()
        status = "ok" if models else "no mlx-lm found"
        print(f"{name:<12} {rt.venv}  ({status})")
        if models:
            extra = sorted(rt.architectures() - base) if name != "mlx" else []
            print(f"{'':<12}   {len(rt.architectures())} architectures"
                  + (f", {len(extra)} not in mlx: {', '.join(extra)}" if extra else ""))
        for key, spec in sorted(rt.flags.items()):
            tools = ", ".join(spec.tools) if spec.tools else "all tools"
            print(f"{'':<12}   {key} -> {spec.flag} ({spec.kind}, {tools})")
