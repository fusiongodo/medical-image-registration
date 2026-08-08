"""Dataset registry: muromi (default) vs acrobat."""

from __future__ import annotations

import json
import os
from pathlib import Path

import conf

DATASETS = ("muromi", "acrobat")
DEFAULT_DATASET = "muromi"

ACROBAT_ROOT = conf.PROJECT_ROOT / "data" / "acrobat"
ACROBAT_RAW = ACROBAT_ROOT / "valid"
ACROBAT_PAIRS = ACROBAT_ROOT / "pairs.json"
ACROBAT_REGWSI = ACROBAT_ROOT / "regwsi"
ACROBAT_RIGID = ACROBAT_ROOT / "rigid"
ACROBAT_POINTS_CSV = ACROBAT_ROOT / "acrobat_validation_points_public_1_of_1.csv"
ACROBAT_ZIP = conf.PROJECT_ROOT / "data" / "valid.zip"


def normalize_dataset(name: str | None) -> str:
    v = (name or DEFAULT_DATASET).strip().lower()
    if v in ("muromi", "mu-romi", "mu_romi"):
        return "muromi"
    if v in ("acrobat", "acro"):
        return "acrobat"
    raise ValueError(f"dataset must be one of {DATASETS}, got {name!r}")


def active_dataset() -> str:
    return normalize_dataset(os.environ.get("MVR_DATASET", DEFAULT_DATASET))


def set_active_dataset(name: str | None) -> str:
    ds = normalize_dataset(name)
    os.environ["MVR_DATASET"] = ds
    return ds


def regwsi_root(dataset: str | None = None) -> Path:
    ds = normalize_dataset(dataset) if dataset else active_dataset()
    if ds == "acrobat":
        return ACROBAT_REGWSI
    return conf.PROJECT_ROOT / "data" / "regwsi"


def rigid_root(dataset: str | None = None) -> Path:
    ds = normalize_dataset(dataset) if dataset else active_dataset()
    if ds == "acrobat":
        return ACROBAT_RIGID
    return conf.PROJECT_ROOT / "data" / "rigid" / "light_v1"


def rigid_path(pair_id: int, dataset: str | None = None) -> Path:
    return rigid_root(dataset) / f"{int(pair_id)}.json"


def pair_dir(pair_id: int, dataset: str | None = None) -> Path:
    return regwsi_root(dataset) / str(int(pair_id))


def load_acrobat_pairs() -> list[dict]:
    if not ACROBAT_PAIRS.is_file():
        return []
    try:
        data = json.loads(ACROBAT_PAIRS.read_text())
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("pairs") or []
    return list(data) if isinstance(data, list) else []


def pair_count(dataset: str | None = None) -> int:
    ds = normalize_dataset(dataset) if dataset else active_dataset()
    if ds == "acrobat":
        return len(load_acrobat_pairs())
    if conf.LABELS_PATH.is_file():
        try:
            labels = json.loads(conf.LABELS_PATH.read_text())
            if isinstance(labels, list):
                return len(labels)
        except Exception:
            pass
    return 0


def pair_fingerprint(pair_id: int, dataset: str | None = None) -> dict:
    ds = normalize_dataset(dataset) if dataset else active_dataset()
    if ds == "acrobat":
        pairs = load_acrobat_pairs()
        if pair_id < 0 or pair_id >= len(pairs):
            return {"dataset": "acrobat", "pair_id": int(pair_id)}
        p = pairs[pair_id]
        return {
            "dataset": "acrobat",
            "pair_id": int(pair_id),
            "case_id": p.get("case_id"),
            "he_file": p.get("he_file"),
            "ihc_file": p.get("ihc_file"),
            "ihc_stain": p.get("ihc_stain"),
        }
    from setup.coarse_to_fine.identity import pair_fingerprint as muromi_fp

    fp = muromi_fp(pair_id)
    return {"dataset": "muromi", **fp}


def cache_namespace(dataset: str | None = None) -> str | None:
    ds = normalize_dataset(dataset) if dataset else active_dataset()
    return None if ds == "muromi" else ds
