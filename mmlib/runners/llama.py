"""llama.cpp's unified `llama` CLI.

Deliberately thin: llama.cpp reads the same Hugging Face cache and already
resolves quants and attaches mmproj vision sidecars on its own, so mm hands it
the short name via -hf and inherits the rest.
"""

from __future__ import annotations

import re
import shutil
import subprocess

NAME = "llama"
FORMATS = ("gguf",)

FLAGS = {
    "llama": {
        "temp": "--temp", "top_p": "--top-p", "top_k": "--top-k",
        "min_p": "--min-p", "max_tokens": "-n", "seed": "--seed",
        "repeat_penalty": "--repeat-penalty",
    },
}


def llama_specs() -> dict[str, str]:
    """Ask llama.cpp which repo:quant handles it will accept, if available."""
    if not shutil.which("llama"):
        return {}
    try:
        out = subprocess.run(["llama", "serve", "--cache-list"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return {}
    specs = {}
    for line in out.splitlines():
        m = re.match(r"\s*\d+\.\s+(\S+/\S+):(\S+)", line)
        if m:
            specs.setdefault(m.group(1).lower(), f"{m.group(1)}:{m.group(2)}")
    return specs


def available() -> str | None:
    return None if shutil.which("llama") else "llama.cpp CLI not on PATH"


def tool_for(entry, serve: bool, args) -> str:
    return "llama"


def command(entry, serve: bool, args, runtime) -> list[str]:
    cmd = ["llama", "serve" if serve else "cli"]
    spec = llama_specs().get(entry.repo.lower()) or entry.hf_spec
    if spec:
        cmd += ["-hf", spec]
    else:
        # No repo:quant to hand over, so point at the file and attach the
        # sidecar ourselves.
        cmd += ["-m", str(entry.path)]
        if entry.mmproj:
            cmd += ["--mmproj", str(entry.mmproj)]
    if serve:
        cmd += ["--port", str(args.port)]
    return cmd
