"""mlx-lm and mlx-vlm, launched out of a registered runtime venv.

The only runner with more than one installation to choose between, which is
what the runtime registry in runtimes.py exists for.
"""

from __future__ import annotations

from ..console import die
from ..runtimes import unsupported_reason

NAME = "mlx"
FORMATS = ("mlx", "hf")

FLAGS = {
    "mlx_lm.chat": {
        "temp": "--temp", "top_p": "--top-p", "max_tokens": "--max-tokens",
        "seed": "--seed", "max_kv_size": "--max-kv-size",
    },
    "mlx_lm.server": {
        "temp": "--temp", "top_p": "--top-p", "top_k": "--top-k",
        "min_p": "--min-p", "max_tokens": "--max-tokens",
    },
    "mlx_vlm.chat": {
        "temp": "--temperature", "max_tokens": "--max-tokens",
        "max_kv_size": "--max-kv-size",
    },
    "mlx_vlm.server": {
        "max_tokens": "--max-tokens",
    },
}


def available() -> str | None:
    """Always: which venv is a per-model question, resolved at launch."""
    return None


def check(entry, runtimes) -> str | None:
    return unsupported_reason(entry, runtimes)


def tool_for(entry, serve: bool, args) -> str:
    vlm = getattr(args, "vlm", False) or (
        entry.role == "vision" and not getattr(args, "lm", False))
    return ("mlx_vlm." if vlm else "mlx_lm.") + ("server" if serve else "chat")


def command(entry, serve: bool, args, runtime) -> list[str]:
    cmd = [runtime.tool(tool_for(entry, serve, args)), "--model", str(entry.path)]
    if serve:
        cmd += ["--port", str(args.port)]
        if args.draft:
            if not entry.draft:
                die(f"no draft model paired with {entry.handle}")
            cmd += ["--draft-model", str(entry.draft)]
    elif args.draft:
        die("mlx_lm.chat has no --draft-model; use `mm serve --draft` instead")
    return cmd
