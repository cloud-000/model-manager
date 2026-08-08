"""Runtimes: the venvs mm can launch an MLX model out of.

There is more than one mlx-lm in the world. Forks add architectures and flags
that upstream does not have, and a checkpoint built against a fork will run on
it and only on it. So the venv is no longer a global constant - it is a named
runtime, models can be pinned to one, and each declares the extra flags it
understands so that `mm set` still refuses typos instead of forwarding them
into a runner that will reject them.

Nothing here runs a subprocess. Everything is read off the filesystem, because
`mm ls` should not pay for a Python interpreter start per runtime.
"""

from __future__ import annotations

import ast
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .console import die
from .paths import DEFAULT_MLX_VENV, expand

# Modules under mlx_lm/models/ that are shared machinery rather than an
# architecture. Harmless in a membership test, but they would inflate the
# counts `mm runtime` reports.
NOT_ARCHITECTURES = {
    "__init__", "base", "cache", "rope_utils", "switch_layers", "activations",
    "bitlinear_layers", "gated_delta", "mla", "pipeline", "ssm",
}


@dataclass
class FlagSpec:
    """One setting a runtime understands that the base vocabulary does not."""
    key: str
    flag: str
    kind: str = "value"           # value | bool
    negatable: bool = False       # bool only: --no-<flag> turns it off explicitly
    tools: tuple[str, ...] = ()   # empty means every tool this runtime provides

    def accepts(self, tool: str) -> bool:
        return not self.tools or tool in self.tools


@dataclass
class Runtime:
    name: str
    venv: Path
    flags: dict[str, FlagSpec] = field(default_factory=dict)
    models_dir_override: Path | None = None
    _archs: dict[str, set[str]] = field(default_factory=dict, repr=False)
    _remap: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)

    # -- launching ---------------------------------------------------------

    def tool(self, name: str) -> str:
        """Prefer this runtime's venv, fall back to PATH."""
        candidate = self.venv / "bin" / name
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(name)
        if not found:
            die(f"{name} not found in {self.venv / 'bin'} or on PATH")
        return found

    def has(self, name: str) -> bool:
        return (self.venv / "bin" / name).is_file()

    # -- introspection -----------------------------------------------------

    def models_dir(self, package: str = "mlx_lm") -> Path | None:
        """Locate this runtime's <package>/models directory.

        An editable install (`pip install -e`) leaves nothing in site-packages
        but a finder shim pointing back at the source tree, which is exactly how
        anyone working on a fork will have installed it.
        """
        if self.models_dir_override and package == "mlx_lm":
            return self.models_dir_override if self.models_dir_override.is_dir() else None
        for sp in sorted(self.venv.glob("lib/python3.*/site-packages")):
            direct = sp / package / "models"
            if direct.is_dir():
                return direct
            for finder in sp.glob("__editable___*_finder.py"):
                try:
                    text = finder.read_text()
                except OSError:
                    continue
                m = re.search(rf"['\"]{package}['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
                if m:
                    candidate = Path(m.group(1)) / "models"
                    if candidate.is_dir():
                        return candidate
        return None

    def architectures(self, package: str = "mlx_lm") -> set[str]:
        """The model_types this runtime ships support for.

        Both packages resolve an architecture by importing
        <package>.models.<model_type> after a remapping table, so the names
        under models/ are the answer. mlx-lm writes one module per
        architecture; mlx-vlm writes one subpackage.
        """
        if package not in self._archs:
            d = self.models_dir(package)
            found: set[str] = set()
            if d is not None:
                found = {p.stem for p in d.glob("*.py")} | {
                    p.name for p in d.iterdir() if (p / "__init__.py").is_file()
                }
                found -= NOT_ARCHITECTURES
            self._archs[package] = found
        return self._archs[package]

    def remapping(self, package: str = "mlx_lm") -> dict[str, str]:
        """The package's MODEL_REMAPPING, read from the installed source.

        Read rather than duplicated: a copy in mm would go stale every time
        upstream aliases a new model_type onto an existing implementation.
        """
        if package not in self._remap:
            table: dict[str, str] = {}
            d = self.models_dir(package)
            utils = d.parent / "utils.py" if d else None
            if utils and utils.is_file():
                try:
                    m = re.search(r"^MODEL_REMAPPING\s*(?::[^=]+)?=\s*(\{.*?^\})",
                                  utils.read_text(), re.S | re.M)
                    if m:
                        value = ast.literal_eval(m.group(1))
                        if isinstance(value, dict):
                            table = value
                except (OSError, ValueError, SyntaxError):
                    pass
            self._remap[package] = table
        return self._remap[package]

    def supports(self, model_type: str | None, package: str = "mlx_lm") -> bool:
        """Can this runtime build this architecture?

        Unknown when the package cannot be introspected at all - an answer of
        "yes" is the safe one there, since refusing to launch a model that would
        have worked is worse than letting the runner report its own error.
        """
        archs = self.architectures(package)
        if not archs or not model_type:
            return True
        return self.remapping(package).get(model_type, model_type) in archs


def parse_flag(key: str, spec) -> FlagSpec:
    if isinstance(spec, str):
        return FlagSpec(key, spec)
    if not isinstance(spec, dict) or "flag" not in spec:
        die(f"runtime flag {key!r} needs a 'flag', got {spec!r}")
    kind = spec.get("type", "value")
    if kind not in ("value", "bool"):
        die(f"runtime flag {key!r}: type must be 'value' or 'bool', got {kind!r}")
    return FlagSpec(key, spec["flag"], kind, bool(spec.get("negatable", False)),
                    tuple(spec.get("tools", ())))


def load_runtimes(cfg: dict) -> dict[str, Runtime]:
    """The registry: always a default `mlx`, plus whatever config adds."""
    from . import runners  # deferred: runners/mlx.py imports this module

    runtimes = {"mlx": Runtime("mlx", DEFAULT_MLX_VENV)}
    reserved = set(runners.NAMES) - {"mlx"}
    for name, spec in (cfg.get("runtimes") or {}).items():
        if name in reserved:
            die(f"{name!r} is the name of a runner and cannot also be a runtime")
        venv = expand(spec["venv"]) if spec.get("venv") else DEFAULT_MLX_VENV
        override = expand(spec["models_dir"]) if spec.get("models_dir") else None
        flags = {k: parse_flag(k, v) for k, v in (spec.get("flags") or {}).items()}
        runtimes[name] = Runtime(name, venv, flags, override)
    return runtimes


def package_for(entry) -> str:
    """Which MLX package will load this: mlx-vlm reads vision models, mlx-lm
    everything else, and the two ship completely different architecture sets."""
    return "mlx_vlm" if entry.role == "vision" else "mlx_lm"


def resolve_runtime(entry, resolved: dict, runtimes: dict[str, Runtime]) -> tuple[Runtime, str]:
    """Pick the runtime for one model, and say why.

    An explicit `runtime=` setting always wins - selection should be something
    you can read out of `mm config`, not something inferred from flags. The
    architecture check is only a backstop for the default.
    """
    pinned = resolved.get("runtime")
    if pinned is not None:
        name = str(pinned[0] if isinstance(pinned, tuple) else pinned)
        if name not in runtimes:
            die(f"unknown runtime {name!r} (known: {', '.join(sorted(runtimes))})")
        return runtimes[name], "pinned"

    pkg = package_for(entry)
    default = runtimes["mlx"]
    if default.supports(entry.model_type, pkg) or entry.custom_code:
        return default, "default"

    for name, rt in runtimes.items():
        if name != "mlx" and rt.supports(entry.model_type, pkg):
            return rt, "only runtime supporting " + str(entry.model_type)
    return default, "default"


def unsupported_reason(entry, runtimes: dict[str, Runtime]) -> str | None:
    """Why no installed runtime can build this architecture, if none can."""
    if entry.fmt not in ("mlx", "hf") or entry.custom_code:
        return None
    pkg = package_for(entry)
    if any(rt.supports(entry.model_type, pkg) for rt in runtimes.values()):
        return None
    name = "mlx-vlm" if pkg == "mlx_vlm" else "mlx-lm"
    return f"{entry.model_type} - no installed {name} implements this architecture"
