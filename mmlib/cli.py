"""Argument parsing and dispatch."""

from __future__ import annotations

import argparse
import sys

from .commands.inventory import cmd_link, cmd_ls, cmd_path
from .commands.meta import cmd_alias, cmd_note, cmd_rm, cmd_root, cmd_runtime
from .commands.settings_cmd import cmd_config, cmd_set
from .config import load_config, migrate_config
from .launch import launch
from .paths import LINK_DIR
from .runtimes import load_runtimes
from .scan import scan
from .settings import scopes, vocab

DESCRIPTION = "a resolver and launcher for locally cached models"


def build_parser(known_scopes, known_keys) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mm", description=DESCRIPTION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ls", help="list cached models")
    s.add_argument("--names", action="store_true", help="handles only (for fzf)")
    s.add_argument("--all", action="store_true", help="include draft models")
    s.set_defaults(func=cmd_ls)

    s = sub.add_parser("path", help="print the resolved runnable path")
    s.add_argument("handle", nargs="?", default=None,
                   help="model handle, alias, or path")
    s.add_argument("--path", type=str, default=None,
                   help="path to a local model directory or weights file")
    s.set_defaults(func=cmd_path)

    for name, serving in (("run", False), ("serve", True)):
        s = sub.add_parser(name, help=f"{'serve' if serving else 'chat with'} a model")
        s.add_argument("handle", nargs="?", default=None,
                       help="model handle, alias, or directory path")
        s.add_argument("--path", type=str, default=None,
                       help="path to a local model directory or weights file")
        s.add_argument("--port", type=int, default=8080)
        s.add_argument("--draft", action="store_true", help="use paired draft model")
        s.add_argument("--vlm", action="store_true", help="force mlx-vlm")
        s.add_argument("--lm", action="store_true", help="force mlx-lm")
        s.add_argument("--dry-run", action="store_true", help="print the command only")
        s.epilog = "anything after a bare -- is forwarded to the runner unchanged"
        s.set_defaults(func=lambda a, e, c, _s=serving: launch(a, e, c, _s))

    s = sub.add_parser("set", help="write settings for a model, runner, or globally")
    s.add_argument("scope", metavar="SCOPE",
                   help=f"a model handle, or one of {', '.join(known_scopes)}")
    s.add_argument("assignments", nargs="*", metavar="KEY=VALUE",
                   help=f"one of: {', '.join(known_keys)}")
    # A single flag with no argument has to be written --raw=--flag: argparse
    # reads a bare '-...' as the next option unless it contains a space.
    s.add_argument("--raw", metavar="FLAGS",
                   help="literal runner flags, e.g. --raw '--xtc-probability 0.1' "
                        "or --raw=--trust-remote-code; pass '' to drop them")
    s.add_argument("--clear", action="append", default=[], metavar="KEY")
    s.set_defaults(func=cmd_set)

    s = sub.add_parser("config", help="show resolved settings and where each came from")
    s.add_argument("handle")
    s.add_argument("--serve", action="store_true",
                   help="resolve against the server tool rather than the chat tool")
    s.add_argument("--vlm", action="store_true")
    s.add_argument("--lm", action="store_true")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("note", help="read or write a note explaining a model")
    s.add_argument("handle")
    s.add_argument("text", nargs="*")
    s.add_argument("--clear", action="store_true")
    s.set_defaults(func=cmd_note)

    s = sub.add_parser("alias", help="pin a short name to a handle")
    s.add_argument("name", nargs="?")
    s.add_argument("handle", nargs="?")
    s.add_argument("--rm", metavar="NAME")
    s.set_defaults(func=cmd_alias)

    s = sub.add_parser("rm", help="delete a cached repo")
    s.add_argument("handle")
    s.add_argument("-y", "--yes", action="store_true")
    s.set_defaults(func=cmd_rm)

    s = sub.add_parser("link", help=f"refresh short-name symlinks in {LINK_DIR}")
    s.set_defaults(func=cmd_link)

    s = sub.add_parser("root", help="directories to scan besides the hub cache")
    r = s.add_subparsers(dest="action")
    r.add_parser("ls", help="list registered roots")
    add = r.add_parser("add", help="register a directory holding models")
    add.add_argument("path")
    rm = r.add_parser("rm", help="stop scanning a directory")
    rm.add_argument("path")
    s.set_defaults(func=cmd_root, action="ls")

    s = sub.add_parser("runtime", help="the venvs MLX models can launch from")
    r = s.add_subparsers(dest="action")
    r.add_parser("ls", help="list runtimes and the architectures each supports")
    add = r.add_parser("add", help="register a venv holding an mlx-lm")
    add.add_argument("name")
    add.add_argument("venv")
    rm = r.add_parser("rm", help="forget a runtime")
    rm.add_argument("name")
    fl = r.add_parser("flag", help="declare a setting only this runtime understands")
    fl.add_argument("name", help="runtime name")
    fl.add_argument("spec", nargs="?", metavar="KEY=--flag",
                    help="e.g. flash_head=--flash-head; omit to list")
    fl.add_argument("--bool", action="store_true",
                    help="the flag takes no value")
    fl.add_argument("--negatable", action="store_true",
                    help="bool only: setting it false emits --no-<flag>")
    fl.add_argument("--tools", metavar="A,B",
                    help="restrict to these tools, e.g. mlx_lm.chat,mlx_lm.server")
    fl.add_argument("--rm", metavar="KEY", help="drop a declared flag")
    s.set_defaults(func=cmd_runtime, action="ls", spec=None, rm=None,
                   tools=None, negatable=False)

    return p


def main() -> None:
    cfg = load_config()
    runtimes = load_runtimes(cfg)
    p = build_parser(scopes(runtimes), vocab(runtimes))

    # Split on a bare -- ourselves. argparse.REMAINDER would happily swallow
    # mm's own flags into the passthrough list.
    argv, extra = sys.argv[1:], []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]

    args = p.parse_args(argv)
    args.extra = extra
    entries = scan(cfg, runtimes=runtimes)
    migrate_config(cfg, entries)
    args.func(args, entries, cfg)
