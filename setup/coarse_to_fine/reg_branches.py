"""Registration config branches: (LAM × field estimator) → curated field-set root."""

from __future__ import annotations

from pathlib import Path

import conf

LAMS = ("fft", "superpoint_glue")
FIELD_ESTIMATORS = ("tps", "wendland", "bspline")
DEFAULT_LAM = "fft"
DEFAULT_FIELD_ESTIMATOR = "tps"
DEFAULT_WENDLAND_EPS = 0.35
DEFAULT_BSPLINE_GRID = 8
DEFAULT_BSPLINE_REG = 1e-3

DATA_ROOT = conf.PROJECT_ROOT / "data"
CURATED_ROOT = DATA_ROOT / "curated_field_sets"
CACHE_ROOT = DATA_ROOT / "c2f_cache"


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


def cache_dir(lam: str | None = None, dataset: str | None = None) -> Path:
    """FFT stays at data/c2f_cache/; other LAMs use data/c2f_cache/{lam}/.

    Non-muromi datasets nest under data/c2f_cache/{dataset}/…
    """
    from setup import datasets as ds

    v = normalize_lam(lam)
    ns = ds.cache_namespace(dataset)
    root = CACHE_ROOT if ns is None else CACHE_ROOT / ns
    if v == "fft":
        return root
    return root / v


def cache_path(
    pair_id: int, depth: int, lam: str | None = None, dataset: str | None = None
) -> Path:
    return cache_dir(lam, dataset=dataset) / f"{int(pair_id)}_d{int(depth)}.json"


def cache_paths(
    pair_id: int, lam: str | None = None, dataset: str | None = None
) -> list[Path]:
    return sorted(cache_dir(lam, dataset=dataset).glob(f"{int(pair_id)}_d*.json"))


def clear_lam_caches(pair_id: int, dataset: str | None = None) -> int:
    """Delete candidate caches for every LAM for this pair (prealignment invalidation)."""
    n = 0
    for lam in LAMS:
        for path in cache_paths(pair_id, lam, dataset=dataset):
            path.unlink()
            n += 1
    return n
