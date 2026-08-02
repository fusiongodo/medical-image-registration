"""Registration config branches: (LAM × field estimator) → curated field-set root."""

from __future__ import annotations

from pathlib import Path

import conf

LAMS = ("fft", "superpoint_glue")
FIELD_ESTIMATORS = ("tps", "wendland")
DEFAULT_LAM = "fft"
DEFAULT_FIELD_ESTIMATOR = "tps"

DATA_ROOT = conf.PROJECT_ROOT / "data"
CURATED_ROOT = DATA_ROOT / "curated_field_sets"


def normalize_lam(lam: str | None) -> str:
    v = (lam or DEFAULT_LAM).strip().lower()
    if v not in LAMS:
        raise ValueError(f"lam must be one of {LAMS}, got {lam!r}")
    return v


def normalize_estimator(field_estimator: str | None) -> str:
    v = (field_estimator or DEFAULT_FIELD_ESTIMATOR).strip().lower()
    if v not in FIELD_ESTIMATORS:
        raise ValueError(f"field_estimator must be one of {FIELD_ESTIMATORS}, got {field_estimator!r}")
    return v


def branch_root(lam: str | None = None, field_estimator: str | None = None) -> Path:
    return CURATED_ROOT / normalize_lam(lam) / normalize_estimator(field_estimator)
