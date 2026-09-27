"""Discover components: every `components/<name>.py` that declares NAME, KIND and build.

A component does nothing until a config switches it on.

    KIND = "feature":  build(frame) -> frame
        `frame` holds dev and test rows (and holdout rows, at audit) with no target column.
        Return it with columns added or changed. Nothing here may learn from labels.
    KIND = "model":    build(params, seed) -> estimator with fit(X, y) and predict_proba(X)
        PARAMS (optional) holds defaults a config can override. `fit` only ever receives
        training-fold rows; any early stopping must split those, never ask for more.
"""
import hashlib
import importlib
import inspect
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable

from harness.paths import ROOT

KINDS = ("feature", "model")


@dataclass(frozen=True)
class Component:
    name: str
    kind: str
    build: Callable
    params: dict = field(default_factory=dict)
    source_hash: str = ""


def discover() -> dict[str, Component]:
    """Files starting with `_` are shared helpers, not components; their source is
    folded into every component's hash so editing a helper invalidates the cache."""
    paths = sorted((ROOT / "components").glob("*.py"))
    helpers = b"".join(p.read_bytes().replace(b"\r\n", b"\n")
                       for p in paths if p.stem.startswith("_"))
    found = {}
    for path in paths:
        if not path.stem.startswith("_"):
            module = importlib.import_module(f"components.{path.stem}")
            found[path.stem] = _load(module, path.stem, helpers)
    return found


def _load(module: ModuleType, stem: str, helpers: bytes) -> Component:
    where = f"components/{stem}.py"
    name = getattr(module, "NAME", None)
    kind = getattr(module, "KIND", None)
    build = getattr(module, "build", None)
    if name != stem:
        raise ValueError(f"{where}: NAME must equal the file name ({stem!r}), got {name!r}")
    if kind not in KINDS:
        raise ValueError(f"{where}: KIND must be one of {KINDS}, got {kind!r}")
    if not callable(build):
        raise ValueError(f"{where}: missing build()")
    params = dict(getattr(module, "PARAMS", {}))
    if kind == "feature" and params:
        raise ValueError(f"{where}: feature components take no PARAMS")
    source = inspect.getsource(module).encode() + helpers
    return Component(name, kind, build, params, hashlib.sha256(source).hexdigest()[:16])
