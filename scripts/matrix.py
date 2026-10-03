"""Score every on/off combination of a batch's screened ideas against the pinned base
(2^k cells for k ideas, the all-off cell being the base itself), skipping cells the planner
declared mutually exclusive. Config-only: each idea must already have a
`configs/exp/<idea>.toml` (written by its phase-1 screen), one line off `main`; a cell
merges the sections those files add (a combiner idea adds a `[combiner]` table; two combiner
ideas are mutually exclusive, so pass them to --exclude) and scores the union with `harness.score`, the normal
way (default seeds 0-2, cached) -- no new components, no re-review.

    python -m scripts.matrix IDEA [IDEA ...] [--exclude IDEA,IDEA] [--exclude ...]
    python -m scripts.matrix IDEA [IDEA ...] --smoke

Prints every cell ranked by CV, highest first, and writes each one's own `results/<run>.json`
via `harness.score.score`, same as any other run. The winning cell (which may be the all-off
cell) is what `scripts.confirm` re-scores against the base next.
"""
import argparse
import itertools
import json
import tomllib
from dataclasses import replace
from pathlib import Path

from harness import config as hconfig
from harness import paths, registry
from harness.score import record, score as harness_score

SECTIONS = ("features", "models", "params", "combiner")
CELL_DIR = paths.CONFIGS / "exp" / "_matrix"
# Cost params capped for --smoke only (never a real matrix run): a cell's CV still costs
# 3 seeds x 5 folds regardless of data size, since these are fixed iteration counts, not
# early-stopped -- same reasoning as scripts/tune.py's own SMOKE_CAPS.
SMOKE_CAPS = {"lgbm": {"n_estimators": 30}, "catboost": {"iterations": 30}}


def read_idea(name: str) -> dict:
    """An idea's own TOML doc (not resolved through `extends`): just what it adds to main."""
    path = paths.CONFIGS / "exp" / f"{name}.toml"
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    if doc.get("extends") != "main":
        raise ValueError(f"{path}: a matrix idea must be `extends = \"main\"` (got {doc.get('extends')!r})")
    return {s: doc.get(s, {}) for s in SECTIONS}


def merge(ideas: dict[str, dict]) -> dict:
    merged = {"features": {}, "models": {}, "params": {}, "combiner": {}}
    for name, doc in ideas.items():
        if doc.get("combiner"):
            if merged["combiner"]:
                raise ValueError(f"idea {name!r} sets [combiner] but another idea in the cell already "
                                 f"does: combiner ideas are mutually exclusive (--exclude them)")
            merged["combiner"] = dict(doc["combiner"])
        merged["features"].update(doc.get("features", {}))
        merged["models"].update(doc.get("models", {}))
        for model, overrides in doc.get("params", {}).items():
            merged["params"].setdefault(model, {}).update(overrides)
    return merged


def write_cell(bits: tuple[str, ...], merged: dict) -> Path:
    lines = ['extends = "main"']
    for section in ("features", "models"):
        if merged[section]:
            lines += [f"[{section}]", *(f"{k} = {json.dumps(v)}" for k, v in merged[section].items())]
    if merged["combiner"]:
        lines += ["[combiner]", *(f"{k} = {json.dumps(v)}" for k, v in merged["combiner"].items())]
    for model, overrides in merged["params"].items():
        if overrides:
            lines += [f"[params.{model}]", *(f"{k} = {json.dumps(v)}" for k, v in overrides.items())]
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    path = CELL_DIR / f"cell-{'+'.join(bits) or 'off'}.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def cells(ideas: list[str], exclude: list[frozenset]) -> list[tuple[str, ...]]:
    """Every subset of `ideas`, all-off first, skipping any subset containing an excluded pair."""
    out = []
    for r in range(len(ideas) + 1):
        for combo in itertools.combinations(ideas, r):
            if any(pair <= set(combo) for pair in exclude):
                continue
            out.append(combo)
    return out


def run_matrix(ideas: list[str], exclude: list[frozenset], caps: bool = False) -> list[dict]:
    comps = registry.discover()
    docs = {name: read_idea(name) for name in ideas}
    rows = []
    for bits in cells(ideas, exclude):
        if not bits:
            cfg = hconfig.load(paths.CONFIGS / "main.toml", comps)
            path = paths.CONFIGS / "main.toml"
        else:
            path = write_cell(bits, merge({n: docs[n] for n in bits}))
            cfg = hconfig.load(path, comps)
        if caps:
            cfg = replace(cfg, params={m: {**p, **SMOKE_CAPS.get(m, {})} for m, p in cfg.params.items()})
        result = harness_score(cfg, comps)
        run_path = record(result)
        rows.append({"cell": bits, "config": str(path), "cv": result["cv"],
                    "delta": result["delta"], "verdict": result["verdict"], "run": run_path.stem})
        print(f"  {'+'.join(bits) or '(off, base)':30s} CV {result['cv']:.5f}  "
              f"delta {result['delta']:+.5f}  verdict {(result['verdict'] or 'base').upper()}")
    return rows


def parse_exclude(raw: list[str]) -> list[frozenset]:
    out = []
    for item in raw or []:
        pair = frozenset(item.split(","))
        if len(pair) != 2:
            raise SystemExit(f"--exclude needs exactly two comma-separated idea names, got {item!r}")
        out.append(pair)
    return out


def smoke() -> None:
    """One of the repo's own example ideas (2 cells: off, on), the real matrix path --
    merge, score, record -- but with model iterations capped (`SMOKE_CAPS`) so it stays
    seconds-long regardless of what `SPACESHIP_DATA` points at."""
    rows = run_matrix(["age_bins"], [], caps=True)
    print(json.dumps({"smoke": "matrix", "cells": len(rows)}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.matrix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ideas", nargs="*", help="idea names, each with a configs/exp/<idea>.toml")
    ap.add_argument("--exclude", action="append", metavar="IDEA,IDEA",
                    help="a pair of ideas never scored together; repeat for more than one pair")
    ap.add_argument("--smoke", action="store_true", help="seconds-long check that the matrix runs (CI)")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()
    if not args.ideas:
        ap.error("at least one idea is required unless --smoke is given")
    if len(args.ideas) > 4:
        ap.error("a batch screens at most 4 ideas; the matrix never sees more")

    rows = run_matrix(args.ideas, parse_exclude(args.exclude))
    winner = max(rows, key=lambda r: r["cv"])
    print(f"\n{len(rows)} cells scored. Winner: {'+'.join(winner['cell']) or '(off, base)'} "
          f"CV {winner['cv']:.5f} ({winner['config']})")
    print(json.dumps({"cells": rows, "winner": winner}))


if __name__ == "__main__":
    main()
