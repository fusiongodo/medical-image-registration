"""
Durable, per-pair global "deskew" store for coarse-to-fine registration.

A deskew is a single rotation-free affine (translation + anisotropic stretch +
shear) fitted from a handful of whole-image correspondence pairs. It is applied
as an image warp of the moving (IHC) channel at crop time (see
setup/live_crop/crop_core.py): every crop of that pair's IHC is resampled
through the affine, so a strong stretch/shear is corrected within each tile
before the FFT/field ever runs. It does NOT touch the displacement field.

Points are stored as normalised [0,1] image fractions, so the preview
resolution used to place them is irrelevant:
    position   (px, py) = (hx, hy)                    in [0,1]^2
    displacement (du,dv) = (hx - ix, hy - iy)         image fractions

Single JSON file per pair at data/deskew/{pair}.json:
    {"pair_id": 3, "identity": {...}, "depth": 1,
     "points": [{"he": [hx, hy], "ihc": [ix, iy]}, ...],   # fractions
     "affine": [[a0, a1, a2], [b0, b1, b2]]}

field_set_cli snapshots/restores this file per field set; crop_core reads only
the `affine`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

from setup.coarse_to_fine.field import Field, fit_affine_norot
from setup.coarse_to_fine.identity import pair_fingerprint

DESKEW_DIR = conf.PROJECT_ROOT / "data" / "deskew"


def _path(pair_id: int) -> Path:
    return DESKEW_DIR / f"{pair_id}.json"


def load(pair_id: int) -> dict | None:
    """Return the stored deskew dict for one pair, or None if absent/unreadable."""
    path = _path(pair_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def clear(pair_id: int) -> None:
    """Remove the stored deskew for one pair (no-op if absent)."""
    path = _path(pair_id)
    if path.exists():
        path.unlink()


def write(pair_id: int, store: dict | None) -> None:
    """Overwrite (or clear, when store is falsy) the raw deskew store for one pair.
    Used by field_set_cli to restore a snapshot."""
    if not store:
        clear(pair_id)
        return
    DESKEW_DIR.mkdir(parents=True, exist_ok=True)
    _path(pair_id).write_text(json.dumps(store, separators=(",", ":")))


def fit(points: list[dict]) -> Field:
    """
    Fit the rotation-free affine field from correspondence pairs.
    points: [{"he": [hx, hy], "ihc": [ix, iy]}, ...] as normalised [0,1] fractions.
    returns: Field(kind="affine").
    """
    pts, du, dv = [], [], []
    for p in points:
        hx, hy = float(p["he"][0]), float(p["he"][1])
        ix, iy = float(p["ihc"][0]), float(p["ihc"][1])
        pts.append([hx, hy])
        du.append(hx - ix)
        dv.append(hy - iy)
    pts = np.asarray(pts, dtype=float)
    w = np.ones(len(points), dtype=float)
    return fit_affine_norot(pts, np.asarray(du), np.asarray(dv), w)


def save(pair_id: int, depth: int, points: list[dict], field: Field) -> Path:
    """Persist the deskew points + fitted affine coefficients for one pair."""
    (au, bu, cu), (av, bv, cv) = field.affine
    store = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "depth": int(depth),
        "points": points,
        "affine": [[au, bu, cu], [av, bv, cv]],
    }
    DESKEW_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(pair_id)
    path.write_text(json.dumps(store, separators=(",", ":")))
    return path
