"""Universal Hugging Face / PyTorch transformers runner.

Runs on Linux, Windows, and macOS (CPU, CUDA, MPS, ROCm).
Used when MLX is not available (e.g. Linux/Windows) or when explicitly selected.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ..console import die
from ..paths import DEFAULT_MLX_VENV, DEFAULT_TRANSFORMERS_VENV

NAME = "transformers"
FORMATS = ("hf",)

FLAGS = {
    "transformers.chat": {
        "temp": "--temp",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "max_tokens": "--max-tokens",
        "seed": "--seed",
    },
    "transformers.server": {
        "temp": "--temp",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "max_tokens": "--max-tokens",
        "seed": "--seed",
    },
}


def find_python() -> Path | None:
    """Find a Python interpreter capable of running transformers."""
    candidates = []
    if "MM_TRANSFORMERS_VENV" in os.environ:
        candidates.append(Path(os.environ["MM_TRANSFORMERS_VENV"]))
    candidates.append(DEFAULT_TRANSFORMERS_VENV)
    if "VIRTUAL_ENV" in os.environ:
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    if DEFAULT_MLX_VENV.is_dir():
        candidates.append(DEFAULT_MLX_VENV)

    for venv in candidates:
        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        py = bin_dir / ("python.exe" if os.name == "nt" else "python")
        if py.is_file():
            return py

    for cmd in (sys.executable, shutil.which("python3"), shutil.which("python")):
        if cmd and Path(cmd).is_file():
            return Path(cmd)
    return None


def available() -> str | None:
    py = find_python()
    if not py:
        return f"python not found for transformers (expected at {DEFAULT_TRANSFORMERS_VENV})"
    return None


def tool_for(entry, serve: bool, args) -> str:
    return "transformers.server" if serve else "transformers.chat"


def command(entry, serve: bool, args, runtime) -> list[str]:
    py = find_python()
    if not py:
        die(f"cannot find python to run transformers (check {DEFAULT_TRANSFORMERS_VENV})")
    script = Path(__file__).parent / ("hf_server.py" if serve else "hf_chat.py")
    cmd = [str(py), str(script), "--model", str(entry.path)]
    if serve:
        cmd += ["--port", str(args.port)]
    return cmd
