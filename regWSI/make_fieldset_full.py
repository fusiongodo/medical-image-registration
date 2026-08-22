"""
Warp IHC into HE space using an FFT field-set (deskew + L5 field) and write
explorer mosaic quads: ihc_fieldset_{tps|wendland}_y{qy}_x{qx}.jpg

Warp matches TRE: at HE pixel p, sample IHC at rigid^{-1}(p - d(p)),
where d = deskew + L5 field in post-rigid HE space.

Usage:
  python regWSI/make_fieldset_full.py <pair_id> --estimator tps
  python regWSI/make_fieldset_full.py <pair_id> --estimator wendland --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import conf
from regWSI import paths
from regWSI.tre_cli import GRID, _resolve_set_dir

JPEG_QUALITY = 85
STRIP_H = 512


def _load_rgb(path: Path) -> np.ndarray:
    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.pages[0].asarray()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _target_size(pair_id: int) -> tuple[int, int, int]:
    meta_path = paths.full_meta_json(pair_id)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return int(meta["w"]), int(meta["h"]), int(meta.get("nq", paths.FULL_NQ))
    return paths.CANVAS_W, paths.CANVAS_H, paths.FULL_NQ


def _layer_name(estimator: str, *, lam: str | None = None, batch_id: str | None = None) -> str:
    if batch_id:
        return f"ihc_eval_{lam}_{estimator}"
    return f"ihc_fieldset_{estimator}"


def _stamp_path(pair_id: int, layer: str) -> Path:
    return paths.full_dir(pair_id) / f"{layer}.stamp.json"


def _field_file(set_dir: Path) -> Path:
    for name in ("field.json", "field_l5.json"):
        p = set_dir / name
        if p.is_file():
            return p
    return set_dir / "field.json"


def _field_mtime(set_dir: Path) -> float:
    field = _field_file(set_dir)
    mt = field.stat().st_mtime if field.is_file() else 0.0
    for name in ("deskew.json", "rigid.json"):
        p = set_dir / name
        if p.is_file():
            mt = max(mt, p.stat().st_mtime)
    return mt


def _load_rigid(set_dir: Path) -> list | None:
    path = set_dir / "rigid.json"
    if not path.is_file():
        return None
    try:
        rigid = json.loads(path.read_text()).get("rigid")
    except Exception:
        return None
    return rigid


def _apply_inv_rigid(
    map_x: np.ndarray, map_y: np.ndarray, rigid, w: int, h: int
) -> tuple[np.ndarray, np.ndarray]:
    (r00, r01, tx), (r10, r11, ty) = rigid
    r = np.array([[float(r00), float(r01)], [float(r10), float(r11)]], dtype=np.float64)
    t = np.array([float(tx), float(ty)], dtype=np.float64)
    rinv = np.linalg.inv(r)
    xn = map_x.astype(np.float64) / w - t[0]
    yn = map_y.astype(np.float64) / h - t[1]
    ix = rinv[0, 0] * xn + rinv[0, 1] * yn
    iy = rinv[1, 0] * xn + rinv[1, 1] * yn
    return (ix * w).astype(np.float32), (iy * h).astype(np.float32)


def _stamp_fresh(pair_id: int, layer: str, set_id: str, set_dir: Path, nq: int) -> bool:
    stamp = _stamp_path(pair_id, layer)
    if not stamp.is_file():
        return False
    try:
        data = json.loads(stamp.read_text())
    except Exception:
        return False
    if data.get("set_id") != set_id:
        return False
    if abs(float(data.get("field_mtime", -1)) - _field_mtime(set_dir)) > 1e-6:
        return False
    for qy in range(nq):
        for qx in range(nq):
            if not paths.full_quadrant(pair_id, layer, qy, qx).is_file():
                return False
    return True


def _dense_disp(set_dir: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (dx, dy) canvas-pixel maps of shape (h, w), HE←IHC displacement."""
    field = json.loads(_field_file(set_dir).read_text())
    depth5 = field.get("depths", {}).get("5") or field.get("depths", {}).get(5) or {}
    scale = w / (GRID * conf.CNN_INPUT_WIDTH)

    dx_l5 = np.zeros((GRID, GRID), dtype=np.float32)
    dy_l5 = np.zeros((GRID, GRID), dtype=np.float32)
    for yi in range(GRID):
        for xi in range(GRID):
            cell = depth5.get(f"{xi}_{yi}") or {}
            dx_l5[yi, xi] = float(cell.get("dx", 0.0)) * scale
            dy_l5[yi, xi] = float(cell.get("dy", 0.0)) * scale

    fdx = cv2.resize(dx_l5, (w, h), interpolation=cv2.INTER_LINEAR)
    fdy = cv2.resize(dy_l5, (w, h), interpolation=cv2.INTER_LINEAR)

    deskew_path = set_dir / "deskew.json"
    if deskew_path.is_file():
        aff = json.loads(deskew_path.read_text()).get("affine")
        if aff is not None:
            (a0, a1, s), (b0, _b1, b2) = aff
            xs = (np.arange(w, dtype=np.float32) + 0.5) / w
            ys = (np.arange(h, dtype=np.float32) + 0.5) / h
            xn, yn = np.meshgrid(xs, ys)
            fdx = fdx + (a0 + a1 * xn + s * yn) * w
            fdy = fdy + (b0 + s * xn + b2 * yn) * h

    return fdx, fdy


def _write_mosaic(img: np.ndarray, pair_id: int, layer: str, nq: int) -> tuple[int, int]:
    h, w = img.shape[:2]
    ys = [round(i * h / nq) for i in range(nq + 1)]
    xs = [round(i * w / nq) for i in range(nq + 1)]
    qw = xs[1] - xs[0]
    qh = ys[1] - ys[0]
    for qy in range(nq):
        for qx in range(nq):
            crop = img[ys[qy] : ys[qy + 1], xs[qx] : xs[qx + 1]]
            out = paths.full_quadrant(pair_id, layer, qy, qx)
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"failed to write {out}")
    return qw, qh


def make_fieldset_full(
    pair_id: int,
    estimator: str,
    force: bool = False,
    *,
    batch_id: str | None = None,
    lam: str = "fft",
) -> dict:
    if estimator not in ("tps", "wendland", "bspline"):
        raise ValueError("estimator must be tps, wendland, or bspline")

    from setup.datasets import ensure_canvas_tiffs

    ensure_canvas_tiffs(pair_id)
    ihc_path = paths.ihc_tiff(pair_id)

    if batch_id:
        from setup.coarse_to_fine import eval_runs

        set_dir = eval_runs.cell_dir(batch_id, pair_id, lam, estimator)
        set_id = f"{batch_id}/{lam}/{estimator}"
        set_name = set_id
        if not _field_file(set_dir).is_file():
            raise FileNotFoundError(f"no field for batch cell {set_id}")
    else:
        set_id, set_dir, set_name = _resolve_set_dir(pair_id, estimator)
        if set_dir is None or not _field_file(set_dir).is_file():
            raise FileNotFoundError(
                f"no field.json for pair {pair_id} fft/{estimator}"
                + (f" (set={set_id})" if set_id else "")
            )

    tw, th, nq = _target_size(pair_id)
    layer = _layer_name(estimator, lam=lam, batch_id=batch_id)
    paths.ensure_pair_dirs(pair_id)

    if not force and _stamp_fresh(pair_id, layer, set_id or "", set_dir, nq):
        print(f"done=1 total=1 cached=1", flush=True)
        return {
            "pair_id": pair_id,
            "estimator": estimator,
            "lam": lam,
            "batch_id": batch_id,
            "layer": layer,
            "set_id": set_id,
            "set_name": set_name,
            "cached": True,
            "w": tw,
            "h": th,
            "nq": nq,
        }

    print(f"done=0 total=4 stage=load", flush=True)
    ihc = _resize_to(_load_rgb(ihc_path), tw, th)
    print(f"done=1 total=4 stage=disp", flush=True)
    dx, dy = _dense_disp(set_dir, tw, th)
    rigid = _load_rigid(set_dir)

    print(f"done=2 total=4 stage=warp", flush=True)
    out = np.empty_like(ihc)
    for y0 in range(0, th, STRIP_H):
        y1 = min(th, y0 + STRIP_H)
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        cols = np.arange(tw, dtype=np.float32)[None, :]
        map_x = (cols - dx[y0:y1]).astype(np.float32)
        map_y = (rows - dy[y0:y1]).astype(np.float32)
        if rigid is not None:
            map_x, map_y = _apply_inv_rigid(map_x, map_y, rigid, tw, th)
        out[y0:y1] = cv2.remap(
            ihc, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        print(f"done=2 total=4 stage=warp y={y1}/{th}", flush=True)
    del ihc, dx, dy

    print(f"done=3 total=4 stage=write", flush=True)
    for old in paths.full_dir(pair_id).glob(f"{layer}_y*_x*.jpg"):
        old.unlink()
    qw, qh = _write_mosaic(out, pair_id, layer, nq)
    del out

    meta_path = paths.full_meta_json(pair_id)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"w": tw, "h": th, "qw": qw, "qh": qh, "nq": nq, "scale": paths.SCALE}
        meta_path.write_text(json.dumps(meta, indent=2))

    stamp = {
        "pair_id": pair_id,
        "estimator": estimator,
        "lam": lam,
        "batch_id": batch_id,
        "set_id": set_id,
        "set_name": set_name,
        "field_mtime": _field_mtime(set_dir),
        "layer": layer,
        "updated": int(time.time()),
    }
    _stamp_path(pair_id, layer).write_text(json.dumps(stamp, separators=(",", ":")))
    print(f"done=4 total=4 stage=done", flush=True)
    return {
        "pair_id": pair_id,
        "estimator": estimator,
        "lam": lam,
        "batch_id": batch_id,
        "layer": layer,
        "set_id": set_id,
        "set_name": set_name,
        "cached": False,
        "w": tw,
        "h": th,
        "nq": nq,
        "qw": qw,
        "qh": qh,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", type=int)
    ap.add_argument("--estimator", required=True, choices=["tps", "wendland", "bspline"])
    ap.add_argument("--lam", default="fft")
    ap.add_argument("--batch", default=None, help="eval batch id (uses eval_runs cell)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(
        json.dumps(
            make_fieldset_full(
                args.pair,
                args.estimator,
                force=args.force,
                batch_id=args.batch,
                lam=args.lam,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
