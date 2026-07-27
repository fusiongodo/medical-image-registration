"""
Phase-correlation translation registration for HE/IHC tile pairs.

Both images are converted to Sobel edge magnitude before registration to
bridge the cross-modal intensity gap between HE and IHC stains.  A Hann window
is applied before the FFT to suppress spectral leakage at tile borders, and
sub-pixel accuracy is obtained by fitting a 2-D parabola around the correlation
peak.

register_arrays() is the in-memory core reused by the coarse-to-fine
orchestrator (setup/coarse_to_fine/run.py).

CLI (level 3, from-scratch): crops tiles live from the raw WSI TIFFs and writes
the per-level FFT displacement field to
    data/c2f_cache/<pair>_d<depth>.json
    {"pair_id", "depth", "levels", "candidates": [
        {"tile_loc", "u", "v", "psr", "delta_px", "by_patch"} ]}

    python align.py <pair_id> <depth>

Sign convention: positive dx shifts IHC rightward relative to HE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))

import crop_core
import tile_metrics

CACHE_DIR = REPO_ROOT / "data" / "c2f_cache"


# ── Core registration ────────────────────────────────────────────────────────

def sobel_edge(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag /= mag.max()
    return mag.astype(np.float32)


def _peak_to_sidelobe(r: np.ndarray, py: int, px: int, exclude: int = 3) -> float:
    """
    Peak-to-sidelobe ratio of a correlation surface: (peak - mean_side) / std_side.
    Sidelobe = all values outside a (2*exclude+1) square centred on the peak.
    """
    h, w = r.shape
    peak = float(r[py, px])
    mask = np.ones_like(r, dtype=bool)
    y0, y1 = max(0, py - exclude), min(h, py + exclude + 1)
    x0, x1 = max(0, px - exclude), min(w, px + exclude + 1)
    mask[y0:y1, x0:x1] = False
    side = r[mask]
    std = float(side.std())
    if std < 1e-12:
        return 0.0
    return (peak - float(side.mean())) / std


def _correlation_surface(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """fftshift-ed normalised cross-power correlation surface of f1 (fixed) vs f2 (moving)."""
    h, w = f1.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F1 = np.fft.fft2(f1 * win)
    F2 = np.fft.fft2(f2 * win)
    R = F1 * np.conj(F2)
    R /= np.abs(R) + 1e-10
    r = np.real(np.fft.ifft2(R))
    return np.fft.fftshift(r)


def _parabolic(arr: np.ndarray, p: int, size: int) -> float:
    if p <= 0 or p >= size - 1:
        return float(p - size // 2)
    pm1, p0, pp1 = arr[p - 1], arr[p], arr[p + 1]
    denom = 2 * p0 - pm1 - pp1
    offset = (pp1 - pm1) / (2 * denom) if abs(denom) > 1e-10 else 0.0
    return float(p - size // 2 + offset)


def _peak_disp(r: np.ndarray, py: int, px: int) -> tuple[float, float, float]:
    h, w = r.shape
    dx = _parabolic(r[py, :], px, w)
    dy = _parabolic(r[:, px], py, h)
    psr = _peak_to_sidelobe(r, int(py), int(px))
    return dx, dy, psr


def phase_correlation(f1: np.ndarray, f2: np.ndarray) -> tuple[float, float, float]:
    """
    Returns (dx, dy, psr): translation IHC must be shifted by to align with HE,
    plus the peak-to-sidelobe ratio of the correlation surface as a confidence.
    Normalised cross-power spectrum in FFT domain; sub-pixel via parabolic fit.
    """
    r = _correlation_surface(f1, f2)
    py, px = np.unravel_index(np.argmax(r), r.shape)
    return _peak_disp(r, int(py), int(px))


def _nms_peaks(r: np.ndarray, n_peaks: int = 5, exclude: int = 3) -> list[tuple[float, float, float]]:
    """
    Non-maximum-suppressed top-N peaks of a precomputed correlation surface.
    Returns a list of (dx, dy, psr) ordered by descending correlation, at most n_peaks.
    Sub-pixel offsets and PSR are read from the original (un-suppressed) surface.
    """
    h, w = r.shape
    work = r.copy()
    out: list[tuple[float, float, float]] = []
    for _ in range(max(1, n_peaks)):
        py, px = np.unravel_index(np.argmax(work), work.shape)
        py, px = int(py), int(px)
        if not np.isfinite(work[py, px]):
            break
        out.append(_peak_disp(r, py, px))
        y0, y1 = max(0, py - exclude), min(h, py + exclude + 1)
        x0, x1 = max(0, px - exclude), min(w, px + exclude + 1)
        work[y0:y1, x0:x1] = -np.inf
    return out


def phase_correlation_multi(
    f1: np.ndarray, f2: np.ndarray, n_peaks: int = 5, exclude: int = 3
) -> list[tuple[float, float, float]]:
    """
    Non-maximum-suppressed top-N peaks of the correlation surface.
    Returns a list of (dx, dy, psr) ordered by descending correlation, at most n_peaks.
    Sub-pixel offsets and PSR are read from the original (un-suppressed) surface.
    """
    r = _correlation_surface(f1, f2)
    return _nms_peaks(r, n_peaks=n_peaks, exclude=exclude)


def register_arrays(he_gray: np.ndarray, ihc_gray: np.ndarray) -> dict[str, float]:
    """
    In-memory registration of two grayscale arrays (HE fixed, IHC moving).
    Returns {dx, dy, psr}. Used by the coarse-to-fine orchestrator on warped tiles.
    """
    fixed = sobel_edge(he_gray.astype(np.float64))
    moving = sobel_edge(ihc_gray.astype(np.float64))
    dx, dy, psr = phase_correlation(fixed, moving)
    return {"dx": dx, "dy": dy, "psr": psr}


def register_arrays_multi(
    he_gray: np.ndarray, ihc_gray: np.ndarray, n_peaks: int = 5
) -> list[dict[str, float]]:
    """
    Multi-peak in-memory registration. Returns up to n_peaks {dx, dy, psr} dicts
    ordered by descending correlation, used by the refinement-aware FFT recompute
    to choose an alternative peak that aligns with the current refinement field.
    """
    fixed = sobel_edge(he_gray.astype(np.float64))
    moving = sobel_edge(ihc_gray.astype(np.float64))
    return [
        {"dx": dx, "dy": dy, "psr": psr}
        for dx, dy, psr in phase_correlation_multi(fixed, moving, n_peaks=n_peaks)
    ]


def surface_and_peaks(
    he_gray: np.ndarray, ihc_gray: np.ndarray, n_peaks: int = 5
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    fftshift-ed correlation surface (H, W) float array plus its NMS top-N peaks.
    Peaks are {dx, dy, psr} ordered by descending correlation. The surface is
    computed once and reused, so this avoids the double FFT of calling the
    surface and the multi-peak helpers separately (used by the FFT map preview).
    """
    fixed = sobel_edge(he_gray.astype(np.float64))
    moving = sobel_edge(ihc_gray.astype(np.float64))
    r = _correlation_surface(fixed, moving)
    peaks = [
        {"dx": dx, "dy": dy, "psr": psr}
        for dx, dy, psr in _nms_peaks(r, n_peaks=n_peaks)
    ]
    return r, peaks


# ── Level-3 from-scratch pass (raw crops -> c2f_cache) ────────────────────────

def process(pair_id: int, depth: int) -> Path:
    tiles = crop_core.tissue_tiles(pair_id, depth)["tiles"]
    total = len(tiles)
    candidates: list[dict] = []

    for done, tile_loc in enumerate(tiles, start=1):
        x, y = (int(p) for p in tile_loc.split("_"))
        he = crop_core.crop_gray(pair_id, depth, x, y, "he")
        ihc = crop_core.crop_gray(pair_id, depth, x, y, "ihc")
        res = register_arrays(he, ihc)
        u, v = res["dx"], res["dy"]
        ihc_auto = crop_core.crop_gray(pair_id, depth, x, y, "ihc", dx=u, dy=v)
        metrics = tile_metrics.tile_metrics(he, ihc, ihc_auto, u, v)
        candidates.append({"tile_loc": tile_loc, "u": u, "v": v, "psr": res["psr"], **metrics})
        print(f"done={done} total={total}", flush=True)

    payload = {"pair_id": pair_id, "depth": depth, "levels": [depth], "candidates": candidates}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{pair_id}_d{depth}.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"pair {pair_id}  depth {depth}: {total} tiles -> {out_path.name}")
    return out_path


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 2:
        sys.exit("Usage: python align.py <pair_id> <depth>")
    process(int(argv[0]), int(argv[1]))


if __name__ == "__main__":
    main()
