"""Google's litert-lm, for `.litertlm` bundles.

The runner that justifies this being Python rather than a table: the model is a
positional argument rather than a flag, and `litert-lm serve` takes no model at
all - it serves everything in its own store at ~/.litert-lm/models and routes
each request by name, the way mlx_lm.server routes on the request's `model`
field.
"""

from __future__ import annotations

import shutil

from ..console import warn

NAME = "litert"
FORMATS = ("litert",)

FLAGS = {
    # --max-num-tokens sizes the KV cache rather than capping the reply, so it
    # is max_kv_size here. litert-lm has no cap-the-reply flag at all.
    "litert-lm.run": {
        "temp": "--temperature", "top_p": "--top-p", "top_k": "--top-k",
        "seed": "--seed", "max_kv_size": "--max-num-tokens",
    },
    # serve accepts --host, --port and --verbose. Nothing else.
    "litert-lm.serve": {},
}


def available() -> str | None:
    return None if shutil.which("litert-lm") else "litert-lm not on PATH"


def tool_for(entry, serve: bool, args) -> str:
    return "litert-lm.serve" if serve else "litert-lm.run"


def command(entry, serve: bool, args, runtime) -> list[str]:
    if serve:
        warn(f"litert-lm serve ignores {entry.handle} and serves its own store "
             f"(litert-lm list); requests route by model name")
        return ["litert-lm", "serve", "--port", str(args.port)]
    if args.draft:
        warn("litert-lm has no --draft-model; speculative decoding is a "
             "property of the bundle (--enable-speculative-decoding)")
    return ["litert-lm", "run", str(entry.path)]
