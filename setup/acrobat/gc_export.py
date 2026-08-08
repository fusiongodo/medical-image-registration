"""
Export ACROBAT validation source landmarks warped by a method for Grand Challenge upload.

Usage:
  python -m setup.acrobat.gc_export --batch <id> --pair 0 --lam fft --estimator tps
  python -m setup.acrobat.gc_export --batch <id> --regwsi --all-pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from regWSI import paths as regwsi_paths
from setup import datasets
from setup.coarse_to_fine import eval_runs, tre_eval


def _load_public_points() -> list[dict]:
    path = datasets.ACROBAT_POINTS_CSV
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _side_fit(meta: dict, side: str) -> dict:
    block = meta.get(side) or {}
    return {
        "scale": float(block.get("scale") or 1.0),
        "offset_x": float(block.get("offset_x") or 0.0),
        "offset_y": float(block.get("offset_y") or 0.0),
        "downsample": float(block.get("downsample") or 1.0),
    }


def _um_to_canvas_xy(x_um: float, y_um: float, mpp: float, fit: dict) -> tuple[float, float]:
    px0 = x_um / mpp
    py0 = y_um / mpp
    ds = max(fit["downsample"], 1e-12)
    px_l = px0 / ds
    py_l = py0 / ds
    return fit["offset_x"] + px_l * fit["scale"], fit["offset_y"] + py_l * fit["scale"]


def _canvas_to_um(x: float, y: float, mpp: float, fit: dict) -> tuple[float, float]:
    ds = max(fit["downsample"], 1e-12)
    scale = max(fit["scale"], 1e-12)
    px_l = (x - fit["offset_x"]) / scale
    py_l = (y - fit["offset_y"]) / scale
    return px_l * ds * mpp, py_l * ds * mpp


def _warp_with_field(
    xn: np.ndarray,
    yn: np.ndarray,
    field_path: Path,
    rigid: list | None,
    w: int,
    h: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    field = json.loads(field_path.read_text())
    if rigid is not None:
        xn, yn = tre_eval.apply_rigid_norm(rigid, xn, yn)
    depth5 = field.get("depths", {}).get("5") or field.get("depths", {}).get(5) or {}
    fdx, fdy = tre_eval.field_disp_canvas(depth5, xn, yn, scale)
    return xn * w + fdx, yn * h + fdy


def _warp_with_df(xn: np.ndarray, yn: np.ndarray, df_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    import SimpleITK as sitk
    from scipy.ndimage import map_coordinates

    img = sitk.ReadImage(str(df_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 2:
        df = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] == 2:
        df = arr
    else:
        raise ValueError(f"unexpected DF shape {arr.shape}")
    _, df_h, df_w = df.shape
    lx = xn * (df_w - 1)
    ly = yn * (df_h - 1)
    ux = map_coordinates(df[0], [ly, lx], order=1, mode="nearest")
    uy = map_coordinates(df[1], [ly, lx], order=1, mode="nearest")
    sx, sy = w / df_w, h / df_h
    return xn * w + ux * sx, yn * h + uy * sy


def export_method(
    *,
    batch_id: str | None,
    pair_ids: list[int],
    lam: str | None,
    estimator: str | None,
    regwsi: bool,
    out_dir: Path,
) -> Path:
    datasets.set_active_dataset("acrobat")
    pairs_index = {int(p["case_id"]): int(p["id"]) for p in datasets.load_acrobat_pairs()}
    rows_in = _load_public_points()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "regwsi" if regwsi else f"{lam}_{estimator}"
    out_path = out_dir / f"registered_landmarks_{tag}.csv"

    fieldnames = [
        "anon_id",
        "anon_filename_he",
        "anon_filename_ihc",
        "point_id",
        "ihc_x",
        "ihc_y",
        "mpp_ihc_10X",
        "mpp_he_10X",
        "ihc_antibody",
        "he_x",
        "he_y",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_in:
            case_id = int(float(row["anon_id"]))
            if case_id not in pairs_index:
                continue
            pair_id = pairs_index[case_id]
            if pair_ids and pair_id not in pair_ids:
                continue
            meta_path = datasets.pair_dir(pair_id, "acrobat") / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            ihc_fit = _side_fit(meta, "ihc")
            he_fit = _side_fit(meta, "he")
            mpp_ihc = float(row.get("mpp_ihc_10X") or 0.91)
            mpp_he = float(row.get("mpp_he_10X") or mpp_ihc)
            cx, cy = _um_to_canvas_xy(float(row["ihc_x"]), float(row["ihc_y"]), mpp_ihc, ihc_fit)
            canvas = meta.get("canvas") or [regwsi_paths.CANVAS_W, regwsi_paths.CANVAS_H]
            cw, ch = int(canvas[0]), int(canvas[1])
            xn = np.array([cx / cw], dtype=float)
            yn = np.array([cy / ch], dtype=float)
            if regwsi:
                df = datasets.pair_dir(pair_id, "acrobat") / "out" / "displacement_field.mha"
                if not df.is_file():
                    continue
                xp, yp = _warp_with_df(xn, yn, df, cw, ch)
            else:
                if not batch_id or not lam or not estimator:
                    raise ValueError("batch/lam/estimator required unless --regwsi")
                field = eval_runs.field_l5_path(batch_id, pair_id, lam, estimator)
                if not field.is_file():
                    continue
                rigid_path = datasets.rigid_path(pair_id, "acrobat")
                rigid = None
                if rigid_path.is_file():
                    rigid = json.loads(rigid_path.read_text()).get("rigid")
                _, _, scale = tre_eval.canvas_scale(pair_id)
                xp, yp = _warp_with_field(xn, yn, field, rigid, cw, ch, scale)
            he_x, he_y = _canvas_to_um(float(xp[0]), float(yp[0]), mpp_he, he_fit)
            out = dict(row)
            out["he_x"] = he_x
            out["he_y"] = he_y
            writer.writerow(out)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--pair", action="append", type=int, default=None)
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--lam", default=None)
    ap.add_argument("--estimator", default=None)
    ap.add_argument("--regwsi", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pairs = list(args.pair or [])
    if args.all_pairs:
        pairs = [int(p["id"]) for p in datasets.load_acrobat_pairs()]
    out = Path(args.out) if args.out else datasets.ACROBAT_ROOT / "gc_exports"
    path = export_method(
        batch_id=args.batch,
        pair_ids=pairs,
        lam=args.lam,
        estimator=args.estimator,
        regwsi=args.regwsi,
        out_dir=out,
    )
    print(json.dumps({"ok": True, "path": str(path)}))


if __name__ == "__main__":
    main()
