"""mlx-lm and mlx-vlm, launched out of a registered runtime venv.

The only runner with more than one installation to choose between, which is
what the runtime registry in runtimes.py exists for.
"""

from __future__ import annotations

import sys

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
    """Always on macOS: which venv is a per-model question, resolved at launch."""
    if sys.platform != "darwin":
        return "mlx only runs on macOS (Apple Silicon)"
    return None


def check(entry, runtimes) -> str | None:
    return unsupported_reason(entry, runtimes)


def tool_for(entry, serve: bool, args) -> str:
    vlm = getattr(args, "vlm", False) or (
        entry.role == "vision" and not getattr(args, "lm", False))
    if serve:
        return "mlx_vlm.server" if vlm else "mlx_lm.server"
    return "mlx_vlm.chat" if vlm else "mlx_lm.chat"


def command(entry, serve: bool, args, runtime) -> list[str]:
    if runtime is None:
        die(f"cannot launch {entry.handle}: no MLX runtime resolved")
    tool = tool_for(entry, serve, args)
    cmd = [runtime.tool(tool), "--model", str(entry.path)]
    if serve:
        cmd += ["--port", str(args.port)]
    if getattr(args, "draft", False) and entry.draft:
        cmd += ["--draft-model", str(entry.draft.path)]
    return cmd
