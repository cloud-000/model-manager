"""llama.cpp's unified `llama` CLI.

Deliberately thin: llama.cpp reads the same Hugging Face cache and already
resolves quants and attaches mmproj vision sidecars on its own, so mm hands it
the short name via -hf and inherits the rest.
"""

from __future__ import annotations

import shutil

NAME = "llama"
FORMATS = ("gguf",)

FLAGS = {
    "llama": {
        "temp": "--temp", "top_p": "--top-p", "top_k": "--top-k",
        "min_p": "--min-p", "max_tokens": "-n", "seed": "--seed",
        "repeat_penalty": "--repeat-penalty",
    },
}


def available() -> str | None:
    return None if shutil.which("llama") else "llama.cpp CLI not on PATH"


def tool_for(entry, serve: bool, args) -> str:
    return "llama"


def command(entry, serve: bool, args, runtime) -> list[str]:
    cmd = ["llama", "serve" if serve else "cli"]
    if entry.hf_spec:
        cmd += ["-hf", entry.hf_spec]
    else:
        # No repo:quant to hand over, so point at the file and attach the
        # sidecar ourselves.
        cmd += ["-m", str(entry.path)]
        if entry.mmproj:
            cmd += ["--mmproj", str(entry.mmproj)]
    if serve:
        cmd += ["--port", str(args.port)]
    return cmd
