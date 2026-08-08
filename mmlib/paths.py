"""Where mm looks for things, and the environment overrides for each."""

from __future__ import annotations

import os
from pathlib import Path

HUB = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache/huggingface/hub")
CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "mm/config.json"
LINK_DIR = Path(os.environ.get("MM_LINK_DIR") or Path.home() / "models")

# The venv backing the default `mlx` runtime. Other runtimes are registered in
# the config file; this one is kept as an environment variable because it
# predates the registry and because it is the one people override per-shell.
DEFAULT_MLX_VENV = Path(os.environ.get("MM_MLX_VENV") or Path.home() / ".venvs/mlx")


def expand(path: str | Path) -> Path:
    """Config files hold ~-relative paths; the rest of mm wants them absolute."""
    return Path(os.path.expanduser(str(path))).resolve()
