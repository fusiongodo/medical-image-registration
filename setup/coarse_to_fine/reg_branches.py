"""Registration config branches: (LAM × field estimator) → curated field-set root."""

from __future__ import annotations

from pathlib import Path

import conf

LAMS = ("fft", "superpoint_glue")
FIELD_ESTIMATORS = ("tps", "wendland", "bspline")
DEFAULT_LAM = "fft"
DEFAULT_FIELD_ESTIMATOR = "tps"
DEFAULT_WENDLAND_EPS_BY_LAM = {"fft": 0.2, "superpoint_glue": 0.1}
DEFAULT_WENDLAND_EPS = DEFAULT_WENDLAND_EPS_BY_LAM["superpoint_glue"]
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


def wendland_eps_for_lam(
    lam: str | None = None,
    wendland_eps: float | None = None,
    by_lam: dict | None = None,
) -> float:
    if wendland_eps is not None:
        return float(wendland_eps)
    merged = {**DEFAULT_WENDLAND_EPS_BY_LAM, **(by_lam or {})}
    try:
        return float(merged[normalize_lam(lam)])
    except (KeyError, ValueError, TypeError):
        return float(DEFAULT_WENDLAND_EPS)


def wendland_eps_tag(wendland_eps: float | None = None, lam: str | None = None) -> str:
    v = (
        float(wendland_eps)
        if wendland_eps is not None
        else wendland_eps_for_lam(lam)
    )
    return f"e{v:g}"


def branch_root(lam: str | None = None, field_estimator: str | None = None) -> Path:
    return CURATED_ROOT / normalize_lam(lam) / normalize_estimator(field_estimator)


def cache_dir(
    lam: str | None = None,
    dataset: str | None = None,
    field_estimator: str | None = None,
    wendland_eps: float | None = None,
) -> Path:
    from setup import datasets as ds

    ns = ds.cache_namespace(dataset)
    root = CACHE_ROOT if ns is None else CACHE_ROOT / ns
    est = normalize_estimator(field_estimator)
    base = root / normalize_lam(lam) / est
    if est != "wendland":
        return base
    return base / wendland_eps_tag(wendland_eps, lam=lam)


def cache_path(
    pair_id: int,
    depth: int,
    lam: str | None = None,
    dataset: str | None = None,
    field_estimator: str | None = None,
    wendland_eps: float | None = None,
) -> Path:
    return cache_dir(
        lam,
        dataset=dataset,
        field_estimator=field_estimator,
        wendland_eps=wendland_eps,
    ) / (f"{int(pair_id)}_d{int(depth)}.json")


def cache_paths(
    pair_id: int,
    lam: str | None = None,
    dataset: str | None = None,
    field_estimator: str | None = None,
    wendland_eps: float | None = None,
) -> list[Path]:
    est = normalize_estimator(field_estimator)
    needle = f"{int(pair_id)}_d*.json"
    if est != "wendland":
        return sorted(
            cache_dir(
                lam, dataset=dataset, field_estimator=est, wendland_eps=wendland_eps
            ).glob(needle)
        )
    from setup import datasets as ds

    ns = ds.cache_namespace(dataset)
    root = CACHE_ROOT if ns is None else CACHE_ROOT / ns
    parent = root / normalize_lam(lam) / est
    found = list(parent.glob(f"e*/{needle}"))
    found.extend(parent.glob(needle))
    return sorted(set(found))


def clear_lam_caches(pair_id: int, dataset: str | None = None) -> int:
    n = 0
    for lam in LAMS:
        for est in FIELD_ESTIMATORS:
            for path in cache_paths(pair_id, lam, dataset=dataset, field_estimator=est):
                path.unlink()
                n += 1
    return n
