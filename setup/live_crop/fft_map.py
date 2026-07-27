"""
Per-tile FFT phase-correlation map preview.

Recrops HE (fixed) and IHC (moving, at the tile's prior base offset) live from
the raw WSI, computes the fftshift-ed phase-correlation surface and its NMS
top-N peaks, and returns a colormapped PNG of the bare surface plus the peak
coordinates so the frontend can overlay fixed-size, toggle-able markers on top.

The whole surface is rendered once at full tile resolution; zoom/pan and marker
rendering happen in the frontend (the peaks cluster near the zero-shift centre).

Peaks / chosen carry image-pixel coords (px, py) alongside the (dx, dy)
displacement: px = dx + w//2, py = dy + h//2. The chosen peak (mx, my) is the
residual that produced the current refinement vector (ux - prior_dx).

fft_map_data(pair, level, tile, dx, dy, mx, my, n_peaks) -> (png_bytes, meta)
"""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "setup" / "auto-alignment"))

import align
import crop_core


def fft_map_data(
    pair: int,
    level: int,
    tile: str,
    dx: float = 0.0,
    dy: float = 0.0,
    mx: float | None = None,
    my: float | None = None,
    n_peaks: int = 5,
) -> tuple[bytes, dict]:
    x, y = (int(p) for p in tile.split("_"))
    he = crop_core.crop_gray(pair, level, x, y, "he")
    ihc = crop_core.crop_gray(pair, level, x, y, "ihc", dx=dx, dy=dy)

    r, peaks = align.surface_and_peaks(he, ihc, n_peaks=n_peaks)
    h, w = r.shape
    cx, cy = w // 2, h // 2

    lo, hi = float(r.min()), float(r.max())
    norm = (r - lo) / (hi - lo) if hi > lo else np.zeros_like(r)
    gray = (norm * 255.0).astype(np.uint8)
    img = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode FFT map PNG")

    peaks_out = [
        {"dx": pk["dx"], "dy": pk["dy"], "psr": pk["psr"], "px": pk["dx"] + cx, "py": pk["dy"] + cy}
        for pk in peaks
    ]
    chosen = None
    if mx is not None and my is not None:
        chosen = {"dx": mx, "dy": my, "px": mx + cx, "py": my + cy}

    meta = {"w": w, "h": h, "cx": cx, "cy": cy, "peaks": peaks_out, "chosen": chosen}
    return buf.tobytes(), meta
