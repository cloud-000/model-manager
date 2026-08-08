"""The settings cascade, and translating it into flags for one runner.

The runners agree on almost nothing: mlx-lm says --temp where mlx-vlm says
--temperature, llama.cpp says -n where everyone else says --max-tokens, and
top-k simply does not exist in the mlx_lm.chat REPL. So settings are written
once in a normalised vocabulary and translated per tool at launch. A key that
the chosen tool cannot express is reported and dropped, never silently eaten.

A runtime may add keys of its own on top of the table - see runtimes.py. The
table itself is assembled from the runner modules, so each runner's vocabulary
sits next to the code that builds its argv.
"""

from __future__ import annotations

from . import runners
from .runtimes import FlagSpec, Runtime, resolve_runtime

FLAG_MAP = runners.flag_map()

# `runtime` selects the venv a model launches from rather than describing a
# sampler, so it travels through the cascade but never becomes a flag.
RESERVED = ("runtime",)

BASE_VOCAB = sorted({k for m in FLAG_MAP.values() for k in m})


def vocab(runtimes: dict[str, Runtime]) -> list[str]:
    keys = set(BASE_VOCAB) | set(RESERVED)
    for rt in runtimes.values():
        keys |= set(rt.flags)
    return sorted(keys)


def scopes(runtimes: dict[str, Runtime]) -> tuple[str, ...]:
    """One scope per runtime, plus one per runner that has no runtimes.

    The MLX runner's scope is the runtime name (@mlx, @deepgrove) because that
    is the thing you would want to set a flag on; every other runner is a
    singleton, so its scope is its own name.
    """
    singletons = tuple(f"@{r.NAME}" for r in runners.RUNNERS if r.NAME != "mlx")
    return ("@defaults",) + tuple(f"@{n}" for n in sorted(runtimes)) + singletons


def flags_for(tool: str, runtime: Runtime | None) -> dict[str, FlagSpec]:
    """Every setting this tool can express, base table plus runtime extras."""
    specs = {k: FlagSpec(k, v) for k, v in FLAG_MAP.get(tool, {}).items()}
    if runtime:
        for key, spec in runtime.flags.items():
            if spec.accepts(tool):
                specs[key] = spec
    return specs


# --------------------------------------------------------------------------
# cascade
# --------------------------------------------------------------------------

def _merge(layers) -> tuple[dict, list[str]]:
    resolved: dict[str, tuple] = {}
    raw: list[str] = []
    for source, layer in layers:
        for key, value in layer.items():
            if key == "raw":
                raw.extend(value)
            else:
                resolved[key] = (value, source)
    return resolved, raw


def pick_runtime(entry, cfg: dict, runtimes: dict[str, Runtime]) -> Runtime | None:
    """Which venv this model launches from, or None for a runner with only one.

    Resolved from the defaults and model layers only: the runner layer is keyed
    by the answer, so consulting it here would be circular.
    """
    runner = runners.for_format(entry.fmt)
    if runner is None or runner.NAME != "mlx":
        return None
    s = cfg["settings"]
    partial, _ = _merge([("defaults", s["defaults"]),
                         ("model", s["model"].get(entry.handle, {}))])
    return resolve_runtime(entry, partial, runtimes)[0]


def runner_name(entry, runtime: Runtime | None) -> str:
    """The settings scope this model's launcher answers to."""
    if runtime is not None:
        return runtime.name
    runner = runners.for_format(entry.fmt)
    return runner.NAME if runner else "mlx"


def resolve_settings(entry, cfg: dict, runner: str) -> tuple[dict, list[str]]:
    """Layer defaults -> runner -> model. Last writer wins per key.

    `raw` is the exception: every layer's raw flags accumulate, because they
    are additional arguments rather than competing values for one setting.
    """
    s = cfg["settings"]
    return _merge([
        ("defaults", s["defaults"]),
        (f"runner:{runner}", s["runner"].get(runner, {})),
        ("model", s["model"].get(entry.handle, {})),
    ])


def settings_flags(resolved: dict, tool: str,
                   runtime: Runtime | None) -> tuple[list[list[str]], list[str]]:
    """Translate resolved settings into flags for one tool; report the misfits.

    Returned as one group per setting rather than a flat list, so that the
    caller can drop a whole group when the user passes the same flag after --.
    A boolean group is one element, a valued group is two.
    """
    specs = flags_for(tool, runtime)
    groups, dropped = [], []
    for key in sorted(resolved):
        if key in RESERVED:
            continue
        value, _ = resolved[key]
        spec = specs.get(key)
        if spec is None:
            dropped.append(key)
        elif spec.kind == "bool":
            if value:
                groups.append([spec.flag])
            elif spec.negatable:
                groups.append(["--no-" + spec.flag.lstrip("-")])
        else:
            groups.append([spec.flag, str(value)])
    return groups, dropped


def parse_value(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text
