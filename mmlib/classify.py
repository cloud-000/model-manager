"""Turning a directory of files into Entry records.

Everything here answers one question: given a path, what is this and can
anything run it? Nothing here knows about handles, runtimes or config.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .entry import Entry

# Quant token at the end of a GGUF filename: Q4_K_M, Q8_0, IQ3_XXS, BF16, F16.
# A vendor prefix such as unsloth's "UD-" is deliberately excluded, because
# llama.cpp's own cache reports the bare quant (UD-Q4_K_XL -> Q4_K_XL).
QUANT_RE = re.compile(r"((?:IQ|Q|BF|F)\d+(?:_[A-Za-z0-9]+)*)$")
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})")


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def dir_size(d: Path) -> int:
    """Size of one snapshot directory, resolving blob symlinks and deduping."""
    total, seen = 0, set()
    for p in d.iterdir():
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        if st.st_ino in seen:
            continue
        seen.add(st.st_ino)
        total += st.st_size
    return total


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def repo_size(repo_dir: Path) -> int:
    blobs = repo_dir / "blobs"
    if not blobs.is_dir():
        return 0
    total = 0
    for p in blobs.iterdir():
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def context_len(cfg: dict) -> int | None:
    for key in ("max_position_embeddings", "num_position_embeddings"):
        if isinstance(cfg.get(key), int):
            return cfg[key]
    text = cfg.get("text_config")
    if isinstance(text, dict):
        return context_len(text)
    return None


def classify_dir(d: Path, repo: str, variant: str | None) -> Entry | None:
    """Classify one directory holding a config.json into a single entry."""
    cfg = read_json(d / "config.json")
    if cfg is None:
        return None

    weights = any(d.glob("*.safetensors"))
    tokenizer = (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()
    model_type = cfg.get("model_type") or "?"

    if not weights:
        # faster-whisper and friends ship CTranslate2 model.bin instead.
        if (d / "model.bin").exists():
            return Entry(repo, "ct2", "none", d, variant, dir_size(d),
                         reason="CTranslate2 weights - needs faster-whisper, not mlx-lm")
        return None

    quant = cfg.get("quantization")
    bits = quant.get("bits") if isinstance(quant, dict) else None
    # A quantization block is proof of MLX; the repo naming is the fallback for
    # unquantised MLX conversions. Anything else is plain safetensors.
    fmt = "mlx" if (bits or "mlx" in repo.lower()) else "hf"

    if not tokenizer:
        return Entry(repo, fmt, "none", d, variant, dir_size(d),
                     model_type=model_type,
                     reason=f"{model_type} - no tokenizer, not a text model")

    role = "draft" if model_type.endswith("_assistant") else \
           "vision" if cfg.get("vision_config") else "chat"

    # A checkpoint naming its own architecture file carries the code to build
    # itself, so no runtime needs to ship support for its model_type.
    custom = bool(cfg.get("model_file") or cfg.get("auto_map"))

    return Entry(repo, fmt, role, d, variant, dir_size(d),
                 quant=f"{bits}bit" if bits else None, ctx=context_len(cfg),
                 model_type=model_type, custom_code=custom)


def classify_weights_dir(d: Path, repo: str, variant: str | None) -> list[Entry]:
    """Every entry a single directory of weights yields: GGUF files, LiteRT
    bundles, and the safetensors layout at the top level."""
    entries: list[Entry] = []

    ggufs = sorted(d.glob("*.gguf"))
    mmproj = next((p for p in ggufs if p.name.lower().startswith("mmproj")), None)
    for p in ggufs:
        if p is mmproj or p.name.lower().startswith("mmproj"):
            continue
        shard = SHARD_RE.search(p.stem)
        if shard and shard.group(1) != "00001":
            continue  # only the first shard of a split model is addressable
        stem = SHARD_RE.sub("", p.stem)
        m = QUANT_RE.search(stem)
        quant = m.group(1) if m else None
        entries.append(Entry(repo, "gguf", "chat", p, quant, file_size(p),
                             quant=quant, mmproj=mmproj,
                             hf_spec=f"{repo}:{quant}" if quant else None))

    # A .litertlm is an opaque bundle - no config.json to read a context length
    # or a quantisation out of. Whether it runs is up to the litert runner.
    for p in d.glob("*.litertlm"):
        entries.append(Entry(repo, "litert", "chat", p, None, file_size(p)))

    root = classify_dir(d, repo, variant)
    if root:
        entries.append(root)
    return entries
