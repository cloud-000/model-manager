"""Finding every set of weights mm knows about, from every source.

Two sources: the Hugging Face hub cache, and any directories registered with
`mm root add`. The second exists because `hf download --local-dir` puts the
weights outside the cache and leaves only a refs/ stub behind, which is how
projects that ship a model alongside their own runtime tend to distribute it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from . import runners
from .classify import classify_dir, classify_weights_dir, dir_size, repo_size
from .console import die
from .entry import Entry
from .paths import HUB, expand
from .runtimes import Runtime, load_runtimes

# Directories that never hold a model but are always sitting next to one.
SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".venv", "tests", "benchmarks"}


def snapshot_dir(repo_dir: Path) -> Path | None:
    """The snapshot `main` points at, falling back to the newest one."""
    ref = repo_dir / "refs/main"
    if ref.is_file():
        candidate = repo_dir / "snapshots" / ref.read_text().strip()
        if candidate.is_dir():
            return candidate
    snaps = repo_dir / "snapshots"
    dirs = [d for d in snaps.iterdir() if d.is_dir()] if snaps.is_dir() else []
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def head_sha(repo_dir: Path) -> str | None:
    ref = repo_dir / "refs/main"
    try:
        return ref.read_text().strip() if ref.is_file() else None
    except OSError:
        return None


def with_variants(d: Path, repo: str, source: str, container: Path) -> list[Entry]:
    """One weights directory plus any quantisation subdirectories under it."""
    entries = classify_weights_dir(d, repo, None)
    for sub in sorted(x for x in d.iterdir() if x.is_dir() and x.name not in SKIP_DIRS):
        child = classify_dir(sub, repo, sub.name)
        if child:
            entries.append(child)
    for e in entries:
        e.source = source
        e.container = container
    return entries


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def scan_hub() -> list[Entry]:
    entries: list[Entry] = []
    for repo_dir in sorted(HUB.glob("models--*")):
        if not repo_dir.is_dir():
            continue
        repo = repo_dir.name[len("models--"):].replace("--", "/", 1)
        snap = snapshot_dir(repo_dir)
        if snap is None:
            # A refs/ stub with no snapshot: `hf download --local-dir` wrote the
            # ref and put the weights somewhere else entirely. Say so rather
            # than silently omitting a repo the user believes they downloaded.
            entries.append(Entry(repo, "hf", "none", repo_dir, None, repo_size(repo_dir),
                                 source="hub", container=repo_dir,
                                 reason="no snapshot in cache - downloaded with "
                                        "--local-dir? (try: mm root add <dir>)"))
            continue
        found = with_variants(snap, repo, "hub", repo_dir)
        if not found:
            found = [Entry(repo, "hf", "none", snap, None, repo_size(repo_dir),
                           source="hub", container=repo_dir,
                           reason="unrecognised layout")]
        entries.extend(found)
    return entries


def hub_index() -> tuple[dict[str, str], dict[str, str]]:
    """Two ways to look up a hub repo id: by commit sha, and by bare name.

    `hf download --local-dir` records the commit in the local directory and the
    ref in the cache, so the two can be rejoined and the real org/name recovered
    instead of guessing an org from a folder name. The name index is the weaker
    fallback for when that metadata has been cleaned away.
    """
    by_sha: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for repo_dir in HUB.glob("models--*"):
        repo = repo_dir.name[len("models--"):].replace("--", "/", 1)
        sha = head_sha(repo_dir)
        if sha:
            by_sha[sha] = repo
        name = repo.split("/", 1)[-1].lower()
        # An ambiguous name is worse than no answer: mark it and never use it.
        by_name[name] = "" if name in by_name and by_name[name] != repo else repo
    return by_sha, {k: v for k, v in by_name.items() if v}


def local_repo_id(d: Path, by_sha: dict[str, str], by_name: dict[str, str]) -> str:
    trees = d / ".cache/huggingface/trees"
    if trees.is_dir():
        for f in trees.glob("*.json"):
            if f.stem in by_sha:
                return by_sha[f.stem]
    return by_name.get(d.name.lower(), f"local/{d.name}")


def scan_roots(roots: list[str]) -> list[Entry]:
    entries: list[Entry] = []
    by_sha, by_name = hub_index()
    for raw in roots:
        root = expand(raw)
        if not root.is_dir():
            continue
        # The root itself may be a model directory, or may merely contain some.
        candidates = [root] + sorted(
            d for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name not in SKIP_DIRS
        )
        for d in candidates:
            has_weights = (d / "config.json").is_file() or any(d.glob("*.gguf"))
            if not has_weights:
                continue
            found = with_variants(d, local_repo_id(d, by_sha, by_name), "local", d)
            for e in found:
                if e.source == "local" and e.size == 0:
                    e.size = dir_size(d)
            entries.extend(found)
    return entries


# --------------------------------------------------------------------------
# naming and cross-references
# --------------------------------------------------------------------------

def make_slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"-(?:gguf|mlx)(?=-|$)", "", s)
    return s


def llama_specs() -> dict[str, str]:
    """Ask llama.cpp which repo:quant handles it will accept, if available."""
    return runners.llama.llama_specs()


def scan(cfg: dict, runtimes: dict[str, Runtime] | None = None) -> list[Entry]:
    if not HUB.is_dir() and not cfg.get("roots"):
        die(f"no Hugging Face cache at {HUB}")

    entries = (scan_hub() if HUB.is_dir() else []) + scan_roots(cfg.get("roots", []))

    # A local copy supersedes the cache stub it was downloaded through: the
    # stub only exists to explain a repo whose weights mm cannot find.
    local_repos = {e.repo for e in entries if e.source == "local"}
    entries = [e for e in entries
               if not (e.source == "hub" and e.repo in local_repos
                       and e.reason and e.reason.startswith("no snapshot"))]

    # Slugs, with the full org-qualified name as a collision tiebreaker.
    # Slugs only have to be unique within their format namespace, so dropping a
    # redundant -MLX/-GGUF suffix is safe: the namespace already carries it.
    # Two repos colliding inside one namespace fall back to org-qualified names.
    by_slug: dict[tuple[str, str], set[str]] = {}
    for e in entries:
        by_slug.setdefault((e.fmt, make_slug(e.repo.split("/", 1)[1])), set()).add(e.repo)
    for e in entries:
        short = make_slug(e.repo.split("/", 1)[1])
        e.slug = short if len(by_slug[(e.fmt, short)]) == 1 \
            else make_slug(e.repo.replace("/", "-"))

    # Pair "<name>-assistant-<...>" draft models with their base model.
    drafts = {re.sub(r"-assistant(?=-|$)", "", e.repo): e
              for e in entries if e.role == "draft"}
    for e in entries:
        if e.fmt in ("mlx", "hf") and e.repo in drafts:
            e.draft = drafts[e.repo].path

    # Last: demote anything no runner can launch, so that every command sees
    # the same answer to "is this runnable" rather than each deciding again.
    runners.annotate(entries, runtimes if runtimes is not None else load_runtimes(cfg))
    return entries


def disk_usage(entries: list[Entry]) -> list[tuple[str, int, Path]]:
    """Bytes per source, for the footer of `mm ls`."""
    rows = []
    if HUB.is_dir():
        rows.append(("hub", sum(repo_size(d) for d in HUB.glob("models--*") if d.is_dir()), HUB))
    seen: set[Path] = set()
    for e in entries:
        if e.source != "local" or e.container is None or e.container in seen:
            continue
        seen.add(e.container)
        rows.append(("local", e.size, e.container))
    return rows
