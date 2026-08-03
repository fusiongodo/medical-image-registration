"""
Compute TRE for regWSI displacement field vs FFT field-set branches (TPS + Wendland).

Landmarks in data/regwsi/{pair}/landmarks.json use normalised [0,1] HE/IHC coords.
Prints one JSON object to stdout.

Usage:
  python regWSI/tre_cli.py <pair_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import conf
from setup.coarse_to_fine.identity import pair_fingerprint, fingerprint_matches
from setup.coarse_to_fine.reg_branches import CURATED_ROOT, FIELD_ESTIMATORS

from regWSI import paths

LEVEL = 5
GRID = 2 ** LEVEL
LAM = "fft"


def _stats(errs: np.ndarray) -> dict:
    if len(errs) == 0:
        return {"mean": None, "median": None, "max": None, "p95": None, "per_point": []}
    return {
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "max": float(np.max(errs)),
        "p95": float(np.percentile(errs, 95)),
        "per_point": [float(e) for e in errs],
    }


def _empty_err(msg: str) -> dict:
    return {
        "mean": None,
        "median": None,
        "max": None,
        "p95": None,
        "per_point": [],
        "error": msg,
    }


def _load_landmarks(pair_id: int) -> list[dict]:
    path = paths.landmarks_json(pair_id)
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    if not fingerprint_matches(pair_id, data.get("identity")):
        raise RuntimeError("landmarks identity does not match current labels")
    return list(data.get("points") or [])


def _branch_root(pair_id: int, field_estimator: str) -> Path:
    return CURATED_ROOT / LAM / field_estimator / str(pair_id)


def _resolve_set_dir(pair_id: int, field_estimator: str) -> tuple[str | None, Path | None, str | None]:
    """
    Pick a set with field.json on this branch.
    Prefer main_set_id, then active set_id, then newest sibling with field.json.
    Returns (set_id, set_dir, set_name).
    """
    root = _branch_root(pair_id, field_estimator)
    if not root.is_dir():
        return None, None, None

    active_path = root / "active.json"
    state: dict = {}
    if active_path.is_file():
        try:
            state = json.loads(active_path.read_text())
        except Exception:
            state = {}

    ordered: list[str] = []
    for key in (state.get("main_set_id"), state.get("set_id")):
        if isinstance(key, str) and key and key not in ordered:
            ordered.append(key)

    def with_field(set_id: str) -> Path | None:
        d = root / set_id
        if d.is_dir() and (d / "field.json").is_file():
            return d
        return None

    for set_id in ordered:
        d = with_field(set_id)
        if d is not None:
            name = _set_name(d, set_id)
            return set_id, d, name

    fallback: list[tuple[float, str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not (child / "field.json").is_file():
            continue
        mtime = (child / "field.json").stat().st_mtime
        fallback.append((mtime, child.name, child))
    if fallback:
        fallback.sort(reverse=True)
        _, set_id, d = fallback[0]
        return set_id, d, _set_name(d, set_id)

    if ordered:
        return ordered[0], None, None
    return None, None, None


def _set_name(set_dir: Path, set_id: str) -> str:
    man = set_dir / "manifest.json"
    if man.is_file():
        try:
            return json.loads(man.read_text()).get("name") or set_id
        except Exception:
            pass
    return set_id


def _deskew_disp_norm(affine, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised HE-minus-IHC displacement from deskew affine at (xn, yn)."""
    (a0, a1, s), (b0, _b1, b2) = affine
    du = a0 + a1 * xn + s * yn
    dv = b0 + s * xn + b2 * yn
    return du, dv


def _field_disp_canvas(field_depth5: dict, xn: np.ndarray, yn: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear-sample L5 tile-centre dx/dy (CNN px) and scale to canvas px.
    xn, yn in [0,1]; returns (dx, dy) in canvas pixels.
    """
    g = GRID
    fx = xn * g - 0.5
    fy = yn * g - 0.5
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    x1 = np.clip(x0 + 1, 0, g - 1)
    y1 = np.clip(y0 + 1, 0, g - 1)
    x0 = np.clip(x0, 0, g - 1)
    y0 = np.clip(y0, 0, g - 1)
    wx = np.clip(fx - x0, 0.0, 1.0)
    wy = np.clip(fy - y0, 0.0, 1.0)

    def at(xi, yi):
        key = f"{int(xi)}_{int(yi)}"
        cell = field_depth5.get(key) or {"dx": 0.0, "dy": 0.0}
        return float(cell["dx"]), float(cell["dy"])

    dx = np.zeros(len(xn), dtype=float)
    dy = np.zeros(len(xn), dtype=float)
    for i in range(len(xn)):
        d00 = at(x0[i], y0[i])
        d10 = at(x1[i], y0[i])
        d01 = at(x0[i], y1[i])
        d11 = at(x1[i], y1[i])
        dx[i] = (
            d00[0] * (1 - wx[i]) * (1 - wy[i])
            + d10[0] * wx[i] * (1 - wy[i])
            + d01[0] * (1 - wx[i]) * wy[i]
            + d11[0] * wx[i] * wy[i]
        )
        dy[i] = (
            d00[1] * (1 - wx[i]) * (1 - wy[i])
            + d10[1] * wx[i] * (1 - wy[i])
            + d01[1] * (1 - wx[i]) * wy[i]
            + d11[1] * wx[i] * wy[i]
        )
    return dx * scale, dy * scale


def _tre_ours(points: list[dict], set_dir: Path, w: int, h: int, scale: float) -> np.ndarray:
    field = json.loads((set_dir / "field.json").read_text())
    depth5 = field.get("depths", {}).get("5") or field.get("depths", {}).get(5) or {}
    deskew_aff = None
    deskew_path = set_dir / "deskew.json"
    if deskew_path.is_file():
        deskew_aff = json.loads(deskew_path.read_text()).get("affine")

    he = np.array([p["he"] for p in points], dtype=float)
    ihc = np.array([p["ihc"] for p in points], dtype=float)
    he_xy = np.stack([he[:, 0] * w, he[:, 1] * h], axis=1)
    ihc_xy = np.stack([ihc[:, 0] * w, ihc[:, 1] * h], axis=1)
    xn, yn = ihc[:, 0], ihc[:, 1]

    du_n = np.zeros(len(points))
    dv_n = np.zeros(len(points))
    if deskew_aff is not None:
        du_n, dv_n = _deskew_disp_norm(deskew_aff, xn, yn)

    fdx, fdy = _field_disp_canvas(depth5, xn, yn, scale)
    dx = du_n * w + fdx
    dy = dv_n * h + fdy
    pred = ihc_xy + np.stack([dx, dy], axis=1)
    return np.linalg.norm(pred - he_xy, axis=1)


def _tre_none(points: list[dict], w: int, h: int) -> np.ndarray:
    he = np.array([p["he"] for p in points], dtype=float)
    ihc = np.array([p["ihc"] for p in points], dtype=float)
    he_xy = np.stack([he[:, 0] * w, he[:, 1] * h], axis=1)
    ihc_xy = np.stack([ihc[:, 0] * w, ihc[:, 1] * h], axis=1)
    return np.linalg.norm(ihc_xy - he_xy, axis=1)


def _tre_regwsi(points: list[dict], pair_id: int, w: int, h: int) -> np.ndarray:
    import SimpleITK as sitk

    df_path = paths.displacement_field(pair_id)
    if not df_path.is_file():
        raise FileNotFoundError(f"missing {df_path}")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(df_path))).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 2:
        df = arr
    elif arr.ndim == 3 and arr.shape[-1] == 2:
        df = np.moveaxis(arr, -1, 0)
    else:
        raise RuntimeError(f"unexpected displacement field shape {arr.shape}")

    _, df_h, df_w = df.shape
    he = np.array([p["he"] for p in points], dtype=float)
    ihc = np.array([p["ihc"] for p in points], dtype=float)
    he_xy = np.stack([he[:, 0] * w, he[:, 1] * h], axis=1)
    lx = ihc[:, 0] * (df_w - 1)
    ly = ihc[:, 1] * (df_h - 1)
    ux = map_coordinates(df[0], [ly, lx], order=1, mode="nearest")
    uy = map_coordinates(df[1], [ly, lx], order=1, mode="nearest")
    sx, sy = w / df_w, h / df_h
    ihc_xy = np.stack([ihc[:, 0] * w, ihc[:, 1] * h], axis=1)
    pred = ihc_xy - np.stack([ux * sx, uy * sy], axis=1)
    return np.linalg.norm(pred - he_xy, axis=1)


def _branch_tre(
    points: list[dict],
    pair_id: int,
    field_estimator: str,
    w: int,
    h: int,
    scale: float,
) -> dict:
    set_id, set_dir, set_name = _resolve_set_dir(pair_id, field_estimator)
    meta = {
        "field_set_id": set_id,
        "field_set_name": set_name,
        "field_estimator": field_estimator,
    }
    if set_dir is None:
        out = _empty_err(
            f"no field.json on fft/{field_estimator}"
            + (f" (main/active={set_id})" if set_id else "")
        )
        out.update(meta)
        return out
    st = _stats(_tre_ours(points, set_dir, w, h, scale))
    st.update(meta)
    return st


def compute_tre(pair_id: int) -> dict:
    points = _load_landmarks(pair_id)
    w, h = paths.CANVAS_W, paths.CANVAS_H
    if paths.full_meta_json(pair_id).is_file():
        meta = json.loads(paths.full_meta_json(pair_id).read_text())
        w, h = int(meta["w"]), int(meta["h"])
    elif paths.meta_json(pair_id).is_file():
        meta = json.loads(paths.meta_json(pair_id).read_text())
        canvas = meta.get("canvas")
        if canvas and len(canvas) == 2:
            w, h = int(canvas[0]), int(canvas[1])

    scale = w / (GRID * conf.CNN_INPUT_WIDTH)

    result: dict = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "n": len(points),
        "canvas": [w, h],
        "scale": scale,
        "tile_w": conf.CNN_INPUT_WIDTH,
        "tile_h": conf.CNN_INPUT_HEIGHT,
    }

    if not points:
        empty = _stats(np.array([]))
        result["none"] = empty
        result["regwsi"] = empty
        result["tps"] = empty
        result["wendland"] = empty
        result["ours"] = empty
        result["field_set_id"] = None
        return result

    result["none"] = _stats(_tre_none(points, w, h))
    result["regwsi"] = _stats(_tre_regwsi(points, pair_id, w, h))

    for est in FIELD_ESTIMATORS:
        result[est] = _branch_tre(points, pair_id, est, w, h, scale)

    # Back-compat: "ours" prefers TPS if present, else Wendland.
    if result["tps"].get("mean") is not None:
        result["ours"] = {**result["tps"]}
        result["field_set_id"] = result["tps"].get("field_set_id")
    elif result["wendland"].get("mean") is not None:
        result["ours"] = {**result["wendland"]}
        result["field_set_id"] = result["wendland"].get("field_set_id")
    else:
        result["ours"] = _empty_err(
            result["tps"].get("error") or result["wendland"].get("error") or "no main field set"
        )
        result["field_set_id"] = None

    for key in ("none", "regwsi", "tps", "wendland", "ours"):
        st = result[key]
        if st.get("mean") is not None:
            st["mean_tile_x"] = st["mean"] / (conf.CNN_INPUT_WIDTH * scale)
            st["mean_tile_y"] = st["mean"] / (conf.CNN_INPUT_HEIGHT * scale)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", type=int)
    args = ap.parse_args()
    print(json.dumps(compute_tre(args.pair), indent=2))


if __name__ == "__main__":
    main()
