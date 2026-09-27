"""Read a config TOML and resolve it against the registry.

    extends = "main"        # optional: start from configs/main.toml, then apply this file
    [features]              # feature components, applied in the order listed
    group = true
    [models]                # model components and their blend weights; 0 = off
    lgbm = 1.0
    [params.lgbm]           # optional overrides of the component's PARAMS
    n_estimators = 400

Anything not listed is off. An experiment config is usually `extends = "main"` plus one line.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path

from harness.paths import CONFIGS, ROOT
from harness.registry import Component

SECTIONS = ("features", "models", "params")


@dataclass(frozen=True)
class Config:
    name: str                       # path relative to the repo, e.g. configs/exp/age_bins.toml
    features: tuple[str, ...]
    weights: dict[str, float]       # enabled models only, weight > 0
    params: dict[str, dict]         # enabled models only: defaults with overrides applied


def load(path: str | Path, registry: dict[str, Component]) -> Config:
    path = Path(path).resolve()
    raw = _read(path, seen=())
    features = tuple(n for n, on in raw["features"].items() if _check_flag(n, on))
    weights = {n: float(w) for n, w in raw["models"].items() if _check_weight(n, w) > 0}
    for n in features:
        _check_kind(registry, n, "feature")
    for n in [*raw["models"], *raw["params"]]:
        _check_kind(registry, n, "model")
    if not weights:
        raise ValueError(f"{path.name}: no model is switched on")
    params = {n: {**registry[n].params, **raw["params"].get(n, {})} for n in weights}
    return Config(_display(path), features, weights, params)


def _read(path: Path, seen: tuple[Path, ...]) -> dict:
    if path in seen:
        raise ValueError(f"extends cycle through {path.name}")
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    unknown = set(doc) - {"extends", *SECTIONS}
    if unknown:
        raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
    merged = {s: {} for s in SECTIONS}
    if "extends" in doc:
        parent = CONFIGS / f"{doc['extends'].removesuffix('.toml')}.toml"
        merged = _read(parent, (*seen, path))
    for s in ("features", "models"):
        merged[s] = {**merged[s], **doc.get(s, {})}
    for model, overrides in doc.get("params", {}).items():
        merged["params"][model] = {**merged["params"].get(model, {}), **overrides}
    return merged


def _check_flag(name: str, on) -> bool:
    if not isinstance(on, bool):
        raise ValueError(f"features.{name} must be true or false, got {on!r}")
    return on


def _check_weight(name: str, w) -> float:
    if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
        raise ValueError(f"models.{name} must be a weight >= 0, got {w!r}")
    return w


def _check_kind(registry: dict[str, Component], name: str, kind: str) -> None:
    if name not in registry:
        raise ValueError(f"no component named {name!r} in components/")
    if registry[name].kind != kind:
        raise ValueError(f"{name!r} is a {registry[name].kind}, not a {kind}")


def _display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
