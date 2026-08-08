"""The record mm builds for every set of weights it can see."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    """Format, role and source are three separate questions.

    `fmt` is how the weights are stored, and it namespaces the handle, so
    mlx/lfm2.5-2.6b:4bit and gguf/lfm2.5-2.6b:Q4_0 can never collide however
    similarly upstream chooses to name their quantisations. `role` is what the
    model is for, which is orthogonal - an MLX repo may hold a chat model, a
    vision model, or a speculative-decoding draft. `source` is where the bytes
    happen to live, which is nobody's business but `mm rm` and `mm link`.
    """
    repo: str                      # "LiquidAI/LFM2.5-2.6B-MLX"
    fmt: str                       # mlx | gguf | hf | litert | ct2
    role: str                      # chat | vision | draft | none
    path: Path                     # what a runner should actually be pointed at
    variant: str | None = None     # "4bit" (MLX subdir) or "Q4_K_M" (GGUF quant)
    size: int = 0
    quant: str | None = None       # bit width for MLX, quant string for GGUF
    ctx: int | None = None
    reason: str | None = None      # why an unrunnable entry cannot run
    hf_spec: str | None = None     # "org/repo:QUANT" for `llama -hf`
    mmproj: Path | None = None     # GGUF vision sidecar, if any
    draft: Path | None = None      # paired speculative-decoding model
    slug: str = ""                 # filled in after collision resolution

    model_type: str | None = None  # config.json's model_type, for runtime checks
    custom_code: bool = False      # checkpoint carries its own architecture file
    source: str = "hub"            # hub | local
    container: Path | None = None  # the directory `mm rm` would delete

    @property
    def handle(self) -> str:
        base = f"{self.fmt}/{self.slug}"
        return f"{base}:{self.variant}" if self.variant else base

    @property
    def runnable(self) -> bool:
        """Whether a runner claimed this is decided during the scan, which
        demotes anything it cannot launch to role 'none' with a reason."""
        return self.role in ("chat", "vision")
