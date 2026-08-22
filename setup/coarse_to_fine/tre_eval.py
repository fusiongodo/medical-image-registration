"""
Shared TRE scoring: landmarks vs identity / regWSI / L5 tile field (+ optional deskew).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

import conf
from setup.coarse_to_fine.identity import fingerprint_matches
from setup.coarse_to_fine.reg_branches import CURATED_ROOT

from regWSI import paths

LEVEL = 5
GRID = 2 ** LEVEL


def stats(errs: np.ndarray) -> dict:
    if len(errs) == 0:
        return {"mean": None, "median": None, "max": None, "p95": None, "per_point": []}
    return {
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "max": float(np.max(errs)),
        "p95": float(np.percentile(errs, 95)),
        "per_point": [float(e) for e in errs],
    }


def empty_err(msg: str) -> dict:
    return {
        "mean": None,
        "median": None,
        "max": None,
        "p95": None,
        "per_point": [],
        "error": msg,
    }


def canvas_size(pair_id: int) -> tuple[int, int]:
    w, h = paths.CANVAS_W, paths.CANVAS_H
    if paths.full_meta_json(pair_id).is_file():
        meta = json.loads(paths.full_meta_json(pair_id).read_text())
        return int(meta["w"]), int(meta["h"])
    if paths.meta_json(pair_id).is_file():
        meta = json.loads(paths.meta_json(pair_id).read_text())
        canvas = meta.get("canvas")
        if canvas and len(canvas) == 2:
            return int(canvas[0]), int(canvas[1])
    return w, h


def canvas_scale(pair_id: int) -> tuple[int, int, float]:
    w, h = canvas_size(pair_id)
    scale = w / (GRID * conf.CNN_INPUT_WIDTH)
    return w, h, scale


def load_landmarks(pair_id: int) -> list[dict]:
    path = paths.landmarks_json(pair_id)
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    from setup import datasets as ds

    if ds.uses_pair_tiffs():
        return list(data.get("points") or [])
    if not fingerprint_matches(pair_id, data.get("identity")):
        raise RuntimeError("landmarks identity does not match current labels")
    return list(data.get("points") or [])


def deskew_disp_norm(affine, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (a0, a1, s), (b0, _b1, b2) = affine
    du = a0 + a1 * xn + s * yn
    dv = b0 + s * xn + b2 * yn
    return du, dv


def apply_rigid_norm(rigid, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    (r00, r01, tx), (r10, r11, ty) = rigid
    xp = float(r00) * xn + float(r01) * yn + float(tx)
    yp = float(r10) * xn + float(r11) * yn + float(ty)
    return xp, yp


def field_disp_canvas(
    field_depth5: dict, xn: np.ndarray, yn: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
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


def _he_xy(points: list[dict], w: int, h: int) -> np.ndarray:
    he = np.array([p["he"] for p in points], dtype=float)
    return np.stack([he[:, 0] * w, he[:, 1] * h], axis=1)


def warp_ihc_none(points: list[dict], w: int, h: int) -> np.ndarray:
    ihc = np.array([p["ihc"] for p in points], dtype=float)
    return np.stack([ihc[:, 0] * w, ihc[:, 1] * h], axis=1)


def xy_norm(xy: np.ndarray, w: int, h: int) -> list[list[float]]:
    if len(xy) == 0:
        return []
    return [[float(p[0] / w), float(p[1] / h)] for p in xy]


def tre_none(points: list[dict], w: int, h: int) -> np.ndarray:
    return np.linalg.norm(warp_ihc_none(points, w, h) - _he_xy(points, w, h), axis=1)


def _load_regwsi_df(pair_id: int) -> np.ndarray:
    import SimpleITK as sitk

    df_path = paths.displacement_field(pair_id)
    if not df_path.is_file():
        raise FileNotFoundError(f"missing {df_path}")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(df_path))).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return np.moveaxis(arr, -1, 0)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr
    raise RuntimeError(f"unexpected displacement field shape {arr.shape}")


def _sample_df(df: np.ndarray, xn: np.ndarray, yn: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    _, df_h, df_w = df.shape
    lx = xn * (df_w - 1)
    ly = yn * (df_h - 1)
    ux = map_coordinates(df[0], [ly, lx], order=1, mode="nearest")
    uy = map_coordinates(df[1], [ly, lx], order=1, mode="nearest")
    return ux, uy, df_w, df_h


def _overlay_ihc_nn(df: np.ndarray, ihc_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    _, df_h, df_w = df.shape
    sx, sy = w / df_w, h / df_h
    ys, xs = np.mgrid[0:df_h, 0:df_w]
    src = np.stack([(xs + df[0]) * sx, (ys + df[1]) * sy], axis=-1).reshape(-1, 2)
    _, idx = cKDTree(src).query(ihc_xy, k=1)
    iy, ix = np.unravel_index(idx, (df_h, df_w))
    return np.stack([ix.astype(float) * sx, iy.astype(float) * sy], axis=1)


def _regwsi_errs_and_overlay(
    points: list[dict], pair_id: int, w: int, h: int
) -> tuple[np.ndarray, np.ndarray]:
    df = _load_regwsi_df(pair_id)
    he = np.array([p["he"] for p in points], dtype=float)
    he_xy = _he_xy(points, w, h)
    ihc_xy = warp_ihc_none(points, w, h)
    ux, uy, df_w, df_h = _sample_df(df, he[:, 0], he[:, 1])
    sx, sy = w / df_w, h / df_h
    sampled = he_xy + np.stack([ux * sx, uy * sy], axis=1)
    errs = np.linalg.norm(sampled - ihc_xy, axis=1)
    q = _overlay_ihc_nn(df, ihc_xy, w, h)
    return errs, q


def tre_regwsi(points: list[dict], pair_id: int, w: int, h: int) -> np.ndarray:
    errs, _ = _regwsi_errs_and_overlay(points, pair_id, w, h)
    return errs


def warp_ihc_regwsi(points: list[dict], pair_id: int, w: int, h: int) -> np.ndarray:
    _, q = _regwsi_errs_and_overlay(points, pair_id, w, h)
    return q


def warp_ihc_field_payload(
    points: list[dict],
    field_payload: dict,
    w: int,
    h: int,
    scale: float,
    deskew_affine=None,
    rigid=None,
) -> np.ndarray:
    depth5 = (
        field_payload.get("depths", {}).get("5")
        or field_payload.get("depths", {}).get(5)
        or {}
    )
    ihc = np.array([p["ihc"] for p in points], dtype=float)
    xn, yn = ihc[:, 0], ihc[:, 1]
    if rigid is not None:
        xn, yn = apply_rigid_norm(rigid, xn, yn)
    base_xy = np.stack([xn * w, yn * h], axis=1)
    du_n = np.zeros(len(points))
    dv_n = np.zeros(len(points))
    if deskew_affine is not None:
        du_n, dv_n = deskew_disp_norm(deskew_affine, xn, yn)
    fdx, fdy = field_disp_canvas(depth5, xn, yn, scale)
    dx = du_n * w + fdx
    dy = dv_n * h + fdy
    return base_xy + np.stack([dx, dy], axis=1)


def tre_field_payload(
    points: list[dict],
    field_payload: dict,
    w: int,
    h: int,
    scale: float,
    deskew_affine=None,
    rigid=None,
) -> np.ndarray:
    pred = warp_ihc_field_payload(
        points, field_payload, w, h, scale, deskew_affine=deskew_affine, rigid=rigid
    )
    return np.linalg.norm(pred - _he_xy(points, w, h), axis=1)


def _field_file_kwargs(
    field_path: Path,
    deskew_path: Path | None = None,
    rigid_path: Path | None = None,
) -> dict:
    field = json.loads(field_path.read_text())
    deskew_aff = None
    if deskew_path is not None and deskew_path.is_file():
        deskew_aff = json.loads(deskew_path.read_text()).get("affine")
    elif (field_path.parent / "deskew.json").is_file():
        deskew_aff = json.loads((field_path.parent / "deskew.json").read_text()).get("affine")
    rigid = None
    if rigid_path is not None and rigid_path.is_file():
        rigid = json.loads(rigid_path.read_text()).get("rigid")
    elif (field_path.parent / "rigid.json").is_file():
        rigid = json.loads((field_path.parent / "rigid.json").read_text()).get("rigid")
    return {"field_payload": field, "deskew_affine": deskew_aff, "rigid": rigid}


def warp_ihc_field_file(
    points: list[dict],
    field_path: Path,
    w: int,
    h: int,
    scale: float,
    deskew_path: Path | None = None,
    rigid_path: Path | None = None,
) -> np.ndarray:
    kw = _field_file_kwargs(field_path, deskew_path, rigid_path)
    return warp_ihc_field_payload(points, w=w, h=h, scale=scale, **kw)


def tre_field_file(
    points: list[dict],
    field_path: Path,
    w: int,
    h: int,
    scale: float,
    deskew_path: Path | None = None,
    rigid_path: Path | None = None,
) -> np.ndarray:
    pred = warp_ihc_field_file(
        points, field_path, w, h, scale, deskew_path=deskew_path, rigid_path=rigid_path
    )
    return np.linalg.norm(pred - _he_xy(points, w, h), axis=1)


def annotate_tile_means(st: dict, scale: float) -> dict:
    if st.get("mean") is not None:
        st = {**st}
        st["mean_tile_x"] = st["mean"] / (conf.CNN_INPUT_WIDTH * scale)
        st["mean_tile_y"] = st["mean"] / (conf.CNN_INPUT_HEIGHT * scale)
    return st


def resolve_curated_set_dir(
    pair_id: int, lam: str, field_estimator: str
) -> tuple[str | None, Path | None, str | None]:
    root = CURATED_ROOT / lam / field_estimator / str(pair_id)
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

    def set_name(set_dir: Path, set_id: str) -> str:
        man = set_dir / "manifest.json"
        if man.is_file():
            try:
                return json.loads(man.read_text()).get("name") or set_id
            except Exception:
                pass
        return set_id

    for set_id in ordered:
        d = with_field(set_id)
        if d is not None:
            return set_id, d, set_name(d, set_id)

    fallback: list[tuple[float, str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not (child / "field.json").is_file():
            continue
        mtime = (child / "field.json").stat().st_mtime
        fallback.append((mtime, child.name, child))
    if fallback:
        fallback.sort(reverse=True)
        _, set_id, d = fallback[0]
        return set_id, d, set_name(d, set_id)

    if ordered:
        return ordered[0], None, None
    return None, None, None


def compute_pair_baseline(pair_id: int) -> dict:
    """none + regwsi TRE (no method matrix)."""
    from setup import datasets as ds

    points = load_landmarks(pair_id)
    w, h, scale = canvas_scale(pair_id)
    result: dict = {
        "pair_id": pair_id,
        "dataset": ds.active_dataset(),
        "identity": ds.pair_fingerprint(pair_id),
        "n": len(points),
        "canvas": [w, h],
        "scale": scale,
        "tile_w": conf.CNN_INPUT_WIDTH,
        "tile_h": conf.CNN_INPUT_HEIGHT,
    }
    if not points:
        empty = stats(np.array([]))
        result["none"] = empty
        result["regwsi"] = empty
        return result

    none_xy = warp_ihc_none(points, w, h)
    result["none"] = annotate_tile_means(
        stats(np.linalg.norm(none_xy - _he_xy(points, w, h), axis=1)), scale
    )
    result["none"]["ihc_warped"] = xy_norm(none_xy, w, h)
    try:
        errs, pred = _regwsi_errs_and_overlay(points, pair_id, w, h)
        result["regwsi"] = annotate_tile_means(stats(errs), scale)
        result["regwsi"]["ihc_warped"] = xy_norm(pred, w, h)
        result["regwsi"]["df_sample"] = "he"
    except Exception as e:
        result["regwsi"] = empty_err(str(e))
    return result
