"""~/.config/mm/config.json: aliases, notes, settings, roots and runtimes.

Nothing about the models themselves is cached here. The scan reads the
filesystem every time, which costs milliseconds and can never go stale.
"""

from __future__ import annotations

import json
from pathlib import Path

from .classify import read_json
from .paths import CONFIG, HUB
from .scan import make_slug


def load_config() -> dict:
    data = read_json(CONFIG) or {}
    for key in ("aliases", "notes", "last_used"):
        data.setdefault(key, {})
    data.setdefault("roots", [])
    data.setdefault("runtimes", {})
    settings = data.setdefault("settings", {})
    for key in ("defaults", "runner", "model"):
        settings.setdefault(key, {})
    return data


def save_config(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def migrate_config(cfg: dict, entries: list) -> None:
    """Rewrite pre-namespace handles ('vibethinker-3b') to 'mlx/vibethinker-3b'.

    Without this, settings and notes keyed by an old handle would keep loading
    but silently never match anything again.
    """
    # Only trust the default cache: a scan of some other HF_HUB_CACHE describes
    # a different set of models and must not rewrite keys based on it.
    if HUB != Path.home() / ".cache/huggingface/hub":
        return

    lookup: dict[str, list[str]] = {}
    for e in entries:
        tail = e.handle.split("/", 1)[1]
        lookup.setdefault(tail, []).append(e.handle)
        # The old scheme fell back to org-qualified slugs on collision, which is
        # precisely the case this namespacing replaced.
        org = make_slug(e.repo.replace("/", "-"))
        legacy = f"{org}:{e.variant}" if e.variant else org
        lookup.setdefault(legacy, []).append(e.handle)

    changed = False
    for section in (cfg["notes"], cfg["last_used"], cfg["settings"]["model"]):
        for old in [k for k in section if "/" not in k]:
            new = lookup.get(old)
            if new and len(new) == 1:
                section[new[0]] = section.pop(old)
                changed = True
    for name, target in cfg["aliases"].items():
        if "/" not in target:
            new = lookup.get(target)
            if new and len(new) == 1:
                cfg["aliases"][name] = new[0]
                changed = True
    if changed:
        save_config(cfg)
