"""
Split HE + raw IHC + warped IHC into an NQxNQ JPEG mosaic for the browser explorer.

Default SCALE=1 canvas is 16384x11008; FULL_NQ=2 → cells ~8192x5504.

Usage:
  python regWSI/make_full.py <pair_id>
  python regWSI/make_full.py <pair_id> --layers ihc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import tifffile

from regWSI import paths

JPEG_QUALITY = 85


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


def _target_size(pair_id: int) -> tuple[int, int, int]:
    """Return (w, h, nq) from existing full/meta or warped IHC, else SCALE defaults."""
    meta_path = paths.full_meta_json(pair_id)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return int(meta["w"]), int(meta["h"]), int(meta.get("nq", paths.FULL_NQ))
    warped = paths.warped_ihc(pair_id)
    if warped.is_file():
        with tifffile.TiffFile(str(warped)) as tif:
            arr = tif.pages[0].asarray()
        h, w = arr.shape[:2]
        return w, h, paths.FULL_NQ
    return paths.CANVAS_W, paths.CANVAS_H, paths.FULL_NQ


def make_full(pair_id: int, layers: tuple[str, ...] | None = None) -> dict:
    layer_list = tuple(layers) if layers else paths.FULL_LAYERS
    for layer in layer_list:
        if layer not in paths.FULL_LAYERS:
            raise ValueError(f"unknown layer {layer}; expected one of {paths.FULL_LAYERS}")

    sources = {
        "he": paths.he_tiff(pair_id),
        "ihc": paths.ihc_tiff(pair_id),
        "ihc_warped": paths.warped_ihc(pair_id),
    }
    for layer in layer_list:
        if not sources[layer].is_file():
            raise FileNotFoundError(f"missing {sources[layer]}")

    tw, th, nq = _target_size(pair_id)
    paths.ensure_pair_dirs(pair_id)

    if layers is None:
        for old in paths.full_dir(pair_id).glob("*.jpg"):
            old.unlink()
    else:
        for layer in layer_list:
            for old in paths.full_dir(pair_id).glob(f"{layer}_y*_x*.jpg"):
                old.unlink()

    qw = qh = 0
    for layer in layer_list:
        img = _load_rgb(sources[layer])
        img = _resize_to(img, tw, th)
        qw, qh = _write_mosaic(img, pair_id, layer, nq)
        del img

    meta = {"w": tw, "h": th, "qw": qw, "qh": qh, "nq": nq, "scale": paths.SCALE}
    paths.full_meta_json(pair_id).write_text(json.dumps(meta, indent=2))
    return {
        "pair_id": pair_id,
        **meta,
        "layers": list(layer_list),
        "dir": str(paths.full_dir(pair_id)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", type=int)
    ap.add_argument(
        "--layers",
        nargs="+",
        choices=list(paths.FULL_LAYERS),
        help="only rebuild these layers (keeps other mosaic files)",
    )
    args = ap.parse_args()
    layers = tuple(args.layers) if args.layers else None
    print(json.dumps(make_full(args.pair, layers=layers), indent=2))


if __name__ == "__main__":
    main()
