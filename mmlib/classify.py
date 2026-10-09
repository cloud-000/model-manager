"""Turning a directory of files into Entry records.

Everything here answers one question: given a path, what is this and can
anything run it? Nothing here knows about handles, runtimes or config.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import NamedTuple

from .entry import Entry

# Quant token at the end of a GGUF filename: Q4_K_M, Q8_0, IQ3_XXS, BF16, F16.
# A vendor prefix such as unsloth's "UD-" is deliberately excluded, because
# llama.cpp's own cache reports the bare quant (UD-Q4_K_XL -> Q4_K_XL).
QUANT_RE = re.compile(r"((?:IQ|Q|BF|F)\d+(?:_[A-Za-z0-9]+)*)$")
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})")

# Directories that never hold a model but are always sitting next to one.
SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", "tests", "benchmarks"}

_REPO_SIZE_CACHE: dict[Path, int] = {}


class DirInfo(NamedTuple):
    has_config: bool
    has_safetensors: bool
    has_tokenizer: bool
    has_model_bin: bool
    size: int
    ggufs: list[tuple[str, Path, int]]
    litert: list[tuple[str, Path, int]]
    subdirs: list[tuple[str, Path]]


def inspect_dir(d: Path) -> DirInfo:
    has_config = False
    has_safetensors = False
    has_tokenizer = False
    has_model_bin = False
    total_size = 0
    seen_inodes: set[int] = set()
    ggufs: list[tuple[str, Path, int]] = []
    litert: list[tuple[str, Path, int]] = []
    subdirs: list[tuple[str, Path]] = []

    try:
        with os.scandir(d) as it:
            for entry in it:
                name = entry.name
                try:
                    if entry.is_dir():
                        if name not in SKIP_DIRS and not name.startswith("."):
                            subdirs.append((name, Path(entry.path)))
                        continue
                except OSError:
                    continue

                if name == "config.json":
                    has_config = True
                elif name in ("tokenizer.json", "tokenizer_config.json"):
                    has_tokenizer = True
                elif name == "model.bin":
                    has_model_bin = True
                elif name.endswith(".safetensors"):
                    has_safetensors = True
                elif name.endswith(".gguf"):
                    try:
                        sz = entry.stat().st_size
                    except OSError:
                        sz = 0
                    ggufs.append((name, Path(entry.path), sz))
                elif name.endswith(".litertlm"):
                    try:
                        sz = entry.stat().st_size
                    except OSError:
                        sz = 0
                    litert.append((name, Path(entry.path), sz))

                try:
                    st = entry.stat()
                    if st.st_ino not in seen_inodes:
                        seen_inodes.add(st.st_ino)
                        total_size += st.st_size
                except OSError:
                    pass
    except OSError:
        pass

    return DirInfo(
        has_config=has_config,
        has_safetensors=has_safetensors,
        has_tokenizer=has_tokenizer,
        has_model_bin=has_model_bin,
        size=total_size,
        ggufs=ggufs,
        litert=litert,
        subdirs=subdirs,
    )


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def dir_size(d: Path) -> int:
    """Size of one snapshot directory, resolving blob symlinks and deduping."""
    total, seen = 0, set()
    try:
        with os.scandir(d) as it:
            for entry in it:
                try:
                    if entry.is_dir():
                        continue
                    st = entry.stat()
                except OSError:
                    continue
                if st.st_ino in seen:
                    continue
                seen.add(st.st_ino)
                total += st.st_size
    except OSError:
        return 0
    return total


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def repo_size(repo_dir: Path) -> int:
    if repo_dir in _REPO_SIZE_CACHE:
        return _REPO_SIZE_CACHE[repo_dir]
    blobs = repo_dir / "blobs"
    total = 0
    try:
        with os.scandir(blobs) as it:
            for entry in it:
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    _REPO_SIZE_CACHE[repo_dir] = total
    return total


def context_len(cfg: dict) -> int | None:
    for key in ("max_position_embeddings", "num_position_embeddings"):
        if isinstance(cfg.get(key), int):
            return cfg[key]
    text = cfg.get("text_config")
    if isinstance(text, dict):
        return context_len(text)
    return None


def classify_dir(d: Path, repo: str, variant: str | None,
                 info: DirInfo | None = None) -> Entry | None:
    """Classify one directory holding a config.json into a single entry."""
    if info is None:
        info = inspect_dir(d)

    if not info.has_config:
        return None

    cfg = read_json(d / "config.json")
    if cfg is None:
        return None

    model_type = cfg.get("model_type") or "?"

    if not info.has_safetensors:
        # faster-whisper and friends ship CTranslate2 model.bin instead.
        if info.has_model_bin:
            return Entry(repo, "ct2", "none", d, variant, info.size,
                         reason="CTranslate2 weights - needs faster-whisper, not mlx-lm")
        return None

    quant = cfg.get("quantization")
    bits = quant.get("bits") if isinstance(quant, dict) else None
    # A quantization block is proof of MLX; the repo naming is the fallback for
    # unquantised MLX conversions. Anything else is plain safetensors.
    fmt = "mlx" if (bits or "mlx" in repo.lower()) else "hf"

    if not info.has_tokenizer:
        return Entry(repo, fmt, "none", d, variant, info.size,
                     model_type=model_type,
                     reason=f"{model_type} - no tokenizer, not a text model")

    role = "draft" if model_type.endswith("_assistant") else \
           "vision" if cfg.get("vision_config") else "chat"

    # A checkpoint naming its own architecture file carries the code to build
    # itself, so no runtime needs to ship support for its model_type.
    custom = bool(cfg.get("model_file") or cfg.get("auto_map"))

    return Entry(repo, fmt, role, d, variant, info.size,
                 quant=f"{bits}bit" if bits else None, ctx=context_len(cfg),
                 model_type=model_type, custom_code=custom)


def classify_weights_dir(d: Path, repo: str, variant: str | None,
                         info: DirInfo | None = None) -> list[Entry]:
    """Every entry a single directory of weights yields: GGUF files, LiteRT
    bundles, and the safetensors layout at the top level."""
    if info is None:
        info = inspect_dir(d)

    entries: list[Entry] = []

    ggufs = sorted(info.ggufs, key=lambda x: x[0])
    mmproj = next((p for name, p, sz in ggufs if name.lower().startswith("mmproj")), None)
    for name, p, sz in ggufs:
        if p is mmproj or name.lower().startswith("mmproj"):
            continue
        shard = SHARD_RE.search(p.stem)
        if shard and shard.group(1) != "00001":
            continue  # only the first shard of a split model is addressable
        stem = SHARD_RE.sub("", p.stem)
        m = QUANT_RE.search(stem)
        quant = m.group(1) if m else None
        entries.append(Entry(repo, "gguf", "chat", p, quant, sz,
                             quant=quant, mmproj=mmproj,
                             hf_spec=f"{repo}:{quant}" if quant else None))

    # A .litertlm is an opaque bundle - no config.json to read a context length
    # or a quantisation out of. Whether it runs is up to the litert runner.
    for name, p, sz in sorted(info.litert, key=lambda x: x[0]):
        entries.append(Entry(repo, "litert", "chat", p, None, sz))

    root = classify_dir(d, repo, variant, info=info)
    if root:
        entries.append(root)
    return entries
