"""
Build browser-friendly preview PNGs from HE + warped IHC at matching geometry.

Caps the longest side at PREVIEW_MAX_SIDE (2048) so the overlay explorer stays light.

Usage:
  python regWSI/make_preview.py <pair_id>
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
from PIL import Image

from regWSI import paths


def _load_rgb(path: Path) -> np.ndarray:
    # DeeperHistReg writes pyramidal BigTIFF; read only the full-res page.
    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.pages[0].asarray()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def make_preview(pair_id: int) -> dict:
    he_path = paths.he_tiff(pair_id)
    warped = paths.warped_ihc(pair_id)
    if not he_path.is_file():
        raise FileNotFoundError(f"missing {he_path}")
    if not warped.is_file():
        raise FileNotFoundError(f"missing {warped}; run register.py first")

    paths.ensure_pair_dirs(pair_id)
    he = _downscale(_load_rgb(he_path), paths.PREVIEW_MAX_SIDE)
    ihc = _downscale(_load_rgb(warped), paths.PREVIEW_MAX_SIDE)
    if he.shape[:2] != ihc.shape[:2]:
        ihc = cv2.resize(ihc, (he.shape[1], he.shape[0]), interpolation=cv2.INTER_AREA)

    he_out = paths.preview_he(pair_id)
    ihc_out = paths.preview_ihc_warped(pair_id)
    Image.fromarray(he, mode="RGB").save(he_out, format="PNG", optimize=True)
    Image.fromarray(ihc, mode="RGB").save(ihc_out, format="PNG", optimize=True)
    return {
        "pair_id": pair_id,
        "w": he.shape[1],
        "h": he.shape[0],
        "he": str(he_out),
        "ihc_warped": str(ihc_out),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", type=int)
    args = ap.parse_args()
    print(json.dumps(make_preview(args.pair), indent=2))


if __name__ == "__main__":
    main()
