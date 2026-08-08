#!/usr/bin/env python3
"""mm - a resolver and launcher for locally cached models.

Turns cached model weights into short, memorable handles and launches each one
with the runtime that can actually load it:

  MLX  -> mlx-lm / mlx-vlm out of a registered runtime venv
  GGUF -> llama.cpp's unified `llama` CLI, delegated via -hf so that
          llama.cpp handles quant selection and mmproj sidecars itself

Stdlib only. No install step, no state beyond ~/.config/mm/config.json.

This file is a launcher: it exists so that `ln -s .../mm ~/.local/bin/mm` keeps
working, resolving the symlink to find the mmlib package next to the real file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mmlib.cli import main  # noqa: E402

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
