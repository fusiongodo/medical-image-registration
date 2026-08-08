"""Eval batch runs under data/eval_runs/{batch_id}/."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import conf
from setup.coarse_to_fine.reg_branches import (
    DEFAULT_BSPLINE_GRID,
    DEFAULT_BSPLINE_REG,
    DEFAULT_WENDLAND_EPS,
    FIELD_ESTIMATORS,
    LAMS,
)

EVAL_ROOT = conf.PROJECT_ROOT / "data" / "eval_runs"
EVAL_DEPTH = 5
DEFAULT_LEVELS = [0, 1, 2, 3, 4, 5]
EXCLUDE_PCT_BY_LEVEL = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.05, 5: 0.10}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str) -> str:
    s = _UNSAFE.sub("-", (name or "").strip()).strip("-").lower()
    return s or f"batch-{int(time.time())}"


def keep_for_level(level: int) -> float:
    excl = float(EXCLUDE_PCT_BY_LEVEL.get(int(level), 0.0))
    return max(0.0, min(1.0, 1.0 - excl))


def default_config() -> dict:
    return {
        "levels": list(DEFAULT_LEVELS),
        "exclude_pct_by_level": dict(EXCLUDE_PCT_BY_LEVEL),
        "wendland_eps": DEFAULT_WENDLAND_EPS,
        "bspline_grid": DEFAULT_BSPLINE_GRID,
        "bspline_reg": DEFAULT_BSPLINE_REG,
        "force": False,
    }


def batch_dir(batch_id: str) -> Path:
    return EVAL_ROOT / batch_id


def manifest_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "manifest.json"


def status_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "status.json"


def cell_dir(batch_id: str, pair_id: int, lam: str, estimator: str) -> Path:
    return batch_dir(batch_id) / str(pair_id) / lam / estimator


def field_l5_path(batch_id: str, pair_id: int, lam: str, estimator: str) -> Path:
    return cell_dir(batch_id, pair_id, lam, estimator) / "field_l5.json"


def tre_path(batch_id: str, pair_id: int, lam: str, estimator: str) -> Path:
    return cell_dir(batch_id, pair_id, lam, estimator) / "tre.json"


def meta_path(batch_id: str, pair_id: int, lam: str, estimator: str) -> Path:
    return cell_dir(batch_id, pair_id, lam, estimator) / "meta.json"


def list_batches() -> list[dict]:
    if not EVAL_ROOT.is_dir():
        return []
    out: list[dict] = []
    for child in sorted(EVAL_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        man = child / "manifest.json"
        if not man.is_file():
            continue
        try:
            m = json.loads(man.read_text())
        except Exception:
            continue
        st = read_status(child.name)
        out.append({**m, "status": st})
    return out


def read_manifest(batch_id: str) -> dict | None:
    path = manifest_path(batch_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_manifest(batch_id: str, manifest: dict) -> Path:
    d = batch_dir(batch_id)
    d.mkdir(parents=True, exist_ok=True)
    path = manifest_path(batch_id)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_status(batch_id: str) -> dict:
    path = status_path(batch_id)
    if not path.is_file():
        return {"state": "idle", "done": 0, "total": 0, "detail": "", "error": None}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"state": "idle", "done": 0, "total": 0, "detail": "", "error": None}


def write_status(batch_id: str, status: dict) -> None:
    d = batch_dir(batch_id)
    d.mkdir(parents=True, exist_ok=True)
    status_path(batch_id).write_text(json.dumps(status, separators=(",", ":")))


def create_batch(
    name: str,
    pairs: list[int],
    *,
    lams: list[str] | None = None,
    estimators: list[str] | None = None,
    config: dict | None = None,
    notes: str | None = None,
    batch_id: str | None = None,
) -> dict:
    bid = batch_id or slugify(name)
    if manifest_path(bid).exists():
        raise FileExistsError(f"batch {bid} already exists")
    cfg = {**default_config(), **(config or {})}
    lam_list = list(lams) if lams else list(LAMS)
    est_list = list(estimators) if estimators else list(FIELD_ESTIMATORS)
    for lam in lam_list:
        if lam not in LAMS:
            raise ValueError(f"unknown lam {lam!r}")
    for est in est_list:
        if est not in FIELD_ESTIMATORS:
            raise ValueError(f"unknown estimator {est!r}")
    manifest = {
        "id": bid,
        "name": name,
        "created": int(time.time()),
        "pairs": [int(p) for p in pairs],
        "lams": lam_list,
        "estimators": est_list,
        "config": cfg,
        "notes": notes or "",
    }
    write_manifest(bid, manifest)
    write_status(
        bid,
        {
            "state": "idle",
            "done": 0,
            "total": len(pairs) * len(lam_list) * len(est_list),
            "detail": "",
            "error": None,
        },
    )
    return manifest


def cell_complete(batch_id: str, pair_id: int, lam: str, estimator: str) -> bool:
    return field_l5_path(batch_id, pair_id, lam, estimator).is_file()
