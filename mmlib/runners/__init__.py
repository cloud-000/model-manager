"""The programs mm can hand a model to, one module each.

A *runner* is a different program with its own calling convention, selected by
the format of the weights. A *runtime* (runtimes.py) is one of several
interchangeable installations of the same runner, selected per model. The two
are not the same thing and merging them does not survive contact with a third
runner: mlx-lm takes the model as `--model <path>`, llama.cpp takes `-hf
<repo>:<quant>` against its own cache, and `litert-lm serve` takes no model
argument whatsoever. Those differences are structural, so they live in Python
rather than in a config file that would have to grow a language to express them.

Each module declares:

    NAME       str                       the settings scope, e.g. @llama
    FORMATS    tuple[str, ...]           Entry.fmt values it claims
    FLAGS      {tool: {key: flag}}       the vocabulary it can express
    available() -> str | None            None, or why it cannot be used
    tool_for(entry, serve, args) -> str  which key of FLAGS applies
    command(entry, serve, args, runtime) -> list[str]

and may declare `check(entry, runtimes) -> str | None` for a per-model reason
it cannot run this particular thing.
"""

from __future__ import annotations

from . import litert, llama, mlx

RUNNERS = (mlx, llama, litert)
NAMES = tuple(r.NAME for r in RUNNERS)


def for_format(fmt: str):
    """The runner that claims this format, if any."""
    for runner in RUNNERS:
        if fmt in runner.FORMATS:
            return runner
    return None


def flag_map() -> dict[str, dict[str, str]]:
    """Every tool every runner knows, assembled.

    Kept next to the code that builds each runner's argv, so adding a runner
    is one file rather than one file and a table somewhere else.
    """
    merged: dict[str, dict[str, str]] = {}
    for runner in RUNNERS:
        merged.update(runner.FLAGS)
    return merged


def check(entry, runtimes) -> str | None:
    """Why this model cannot run here, or None if it can."""
    if entry.role not in ("chat", "vision"):
        return entry.reason
    runner = for_format(entry.fmt)
    if runner is None:
        return entry.reason or f"{entry.fmt} - no runner handles this format"
    return runner.available() or (
        runner.check(entry, runtimes) if hasattr(runner, "check") else None)


def annotate(entries, runtimes) -> None:
    """Demote everything that cannot actually be launched, with the reason.

    Listing a model as runnable and then failing at load time is the one thing
    the "not runnable here" section exists to prevent, so this runs during the
    scan and every command sees the same answer.
    """
    for entry in entries:
        reason = check(entry, runtimes)
        if reason and entry.role in ("chat", "vision"):
            entry.role, entry.reason = "none", reason
