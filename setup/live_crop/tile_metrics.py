"""
LNCC-squared tile metrics on in-memory grayscale arrays.

Extracted from setup/auto-alignment/svelte_metrics.py so the coarse-to-fine
alignment can fold per-tile metrics into c2f_cache directly (no per-tile
metrics.json sidecars). Normalisation and the summed-area-table LNCC mirror
computeLNCC in svelte-app imageUtils.ts (target mean 128, std 64).

normalize_gray(arr)                              -> float32 (H, W)
compute_lncc2(g1, g2, patch_size)                -> float in [0, 1]
tile_metrics(he, ihc_base, ihc_auto, u, v)       -> {delta_px, by_patch}
"""

from __future__ import annotations

import math

import numpy as np

PATCH_SIZES = [5, 10, 20, 30, 40, 50]


def normalize_gray(arr: np.ndarray) -> np.ndarray:
    g = arr.astype(np.float64)
    mean = g.mean()
    std = g.std() or 1.0
    return np.clip((g - mean) / std * 64.0 + 128.0, 0, 255).astype(np.float32)


def _build_sat(arr: np.ndarray) -> np.ndarray:
    sat = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)
    return sat


def compute_lncc2(g1: np.ndarray, g2: np.ndarray, patch_size: int) -> float:
    r = patch_size // 2
    area = float(patch_size * patch_size)
    h, w = g1.shape

    sat1 = _build_sat(g1)
    sat2 = _build_sat(g2)
    sat1sq = _build_sat(g1.astype(np.float64) ** 2)
    sat2sq = _build_sat(g2.astype(np.float64) ** 2)
    sat12 = _build_sat(g1.astype(np.float64) * g2.astype(np.float64))

    ys = np.arange(r, h - r)
    xs = np.arange(r, w - r)
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    def rs(sat: np.ndarray) -> np.ndarray:
        return (sat[Y + r + 1, X + r + 1]
                - sat[Y - r,     X + r + 1]
                - sat[Y + r + 1, X - r    ]
                + sat[Y - r,     X - r    ])

    s1 = rs(sat1)
    s2 = rs(sat2)
    s1sq = rs(sat1sq)
    s2sq = rs(sat2sq)
    s12 = rs(sat12)

    mu1 = s1 / area
    mu2 = s2 / area
    num = s12 - area * mu1 * mu2
    den1 = np.maximum(0.0, s1sq - area * mu1 ** 2)
    den2 = np.maximum(0.0, s2sq - area * mu2 ** 2)
    den = den1 * den2

    valid = den > 1e-6
    scores = np.where(valid, num ** 2 / np.where(valid, den, 1.0), 0.0)
    n = int(valid.sum())
    return float(scores.sum() / n) if n > 0 else 0.0


def tile_metrics(
    he: np.ndarray,
    ihc_base: np.ndarray,
    ihc_auto: np.ndarray,
    u: float,
    v: float,
) -> dict:
    """
    he / ihc_base / ihc_auto : grayscale arrays (same H, W). ihc_base is the
    moving tile at the level's base offset (0 at L3, saved field at L4/5);
    ihc_auto is the moving tile at the composed auto displacement (u, v).
    Returns {delta_px, by_patch{ps: {lncc2, lncc2_auto, factor_auto}}}.
    """
    g_he = normalize_gray(he)
    g_base = normalize_gray(ihc_base)
    g_auto = normalize_gray(ihc_auto)

    by_patch: dict[str, dict[str, float]] = {}
    for ps in PATCH_SIZES:
        lncc2_base = compute_lncc2(g_he, g_base, ps)
        lncc2_auto = compute_lncc2(g_he, g_auto, ps)
        factor = lncc2_auto / lncc2_base if lncc2_base > 1e-9 else 0.0
        by_patch[str(ps)] = {
            "lncc2": lncc2_base,
            "lncc2_auto": lncc2_auto,
            "factor_auto": factor,
        }

    return {"delta_px": math.sqrt(u ** 2 + v ** 2), "by_patch": by_patch}
