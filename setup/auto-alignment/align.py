"""
Phase-correlation translation registration for HE/IHC tile pairs.

Both images are converted to Sobel edge magnitude before registration to
bridge the cross-modal intensity gap between HE and IHC stains.
A Hann window is applied before the FFT to suppress spectral leakage at tile
borders, and sub-pixel accuracy is obtained by fitting a 2-D parabola around
the correlation peak.

Usage — single depth:
    python align.py <pair_id> <depth> [tile_id ...] [--force]

Usage — batch (next N unprocessed pairs, all depths):
    python align.py --pairs N [--force]

    A pair is "unprocessed" when it contains no elastix/displacement.json
    files across any of its depth directories.  Pairs are processed in
    ascending numeric order.

Output per tile (idempotent unless --force):
    data/cropped/<pair_id>/d<depth>/<tile_id>/elastix/displacement.json
    {"dx": <float>, "dy": <float>}

Sign convention: positive dx shifts IHC rightward relative to HE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "cropped"
RESULT_FILENAME = "displacement.json"
ALG_DIR = "elastix"


# ── Core registration ────────────────────────────────────────────────────────

def sobel_edge(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag /= mag.max()
    return mag.astype(np.float32)


def phase_correlation(f1: np.ndarray, f2: np.ndarray) -> tuple[float, float]:
    """
    Returns (dx, dy): translation IHC must be shifted by to align with HE.
    Normalised cross-power spectrum in FFT domain; sub-pixel via parabolic fit.
    """
    h, w = f1.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F1 = np.fft.fft2(f1 * win)
    F2 = np.fft.fft2(f2 * win)
    R = F1 * np.conj(F2)
    R /= np.abs(R) + 1e-10
    r = np.real(np.fft.ifft2(R))
    r = np.fft.fftshift(r)

    py, px = np.unravel_index(np.argmax(r), r.shape)

    def parabolic(arr: np.ndarray, p: int, size: int) -> float:
        if p <= 0 or p >= size - 1:
            return float(p - size // 2)
        pm1, p0, pp1 = arr[p - 1], arr[p], arr[p + 1]
        denom = 2 * p0 - pm1 - pp1
        offset = (pp1 - pm1) / (2 * denom) if abs(denom) > 1e-10 else 0.0
        return float(p - size // 2 + offset)

    return parabolic(r[py, :], px, w), parabolic(r[:, px], py, h)


def register_tile(tile_dir: Path) -> dict[str, float]:
    he  = cv2.imread(str(tile_dir / "he.png"),  cv2.IMREAD_GRAYSCALE)
    ihc = cv2.imread(str(tile_dir / "ihc.png"), cv2.IMREAD_GRAYSCALE)
    fixed  = sobel_edge(he.astype(np.float64))
    moving = sobel_edge(ihc.astype(np.float64))
    dx, dy = phase_correlation(fixed, moving)
    return {"dx": dx, "dy": dy}


# ── Processing helpers ───────────────────────────────────────────────────────

def process_depth(depth_dir: Path, force: bool) -> tuple[int, int]:
    """Process all tiles in one depth directory. Returns (done, skipped)."""
    done = skipped = 0
    tiles = sorted(d for d in depth_dir.iterdir() if d.is_dir())
    for tile_dir in tiles:
        out_file = tile_dir / ALG_DIR / RESULT_FILENAME
        if out_file.exists() and not force:
            skipped += 1
            continue
        if not (tile_dir / "he.png").exists() or not (tile_dir / "ihc.png").exists():
            skipped += 1
            continue
        try:
            result = register_tile(tile_dir)
        except Exception as exc:
            print(f"    ERROR {tile_dir.name}: {exc}")
            skipped += 1
            continue
        (tile_dir / ALG_DIR).mkdir(exist_ok=True)
        out_file.write_text(json.dumps(result))
        done += 1
    return done, skipped


def pair_is_processed(pair_dir: Path) -> bool:
    return any(pair_dir.rglob(f"{ALG_DIR}/{RESULT_FILENAME}"))


def process(pair_id: int, depth: int, tile_ids: list[str], force: bool) -> None:
    depth_dir = DATA_ROOT / str(pair_id) / f"d{depth}"
    if not depth_dir.is_dir():
        sys.exit(f"Directory not found: {depth_dir}")

    tiles = (
        [depth_dir / t for t in tile_ids]
        if tile_ids
        else sorted(d for d in depth_dir.iterdir() if d.is_dir())
    )

    for tile_dir in tiles:
        out_file = tile_dir / ALG_DIR / RESULT_FILENAME
        if out_file.exists() and not force:
            print(f"skip  {tile_dir.name}  (already computed)")
            continue
        if not (tile_dir / "he.png").exists() or not (tile_dir / "ihc.png").exists():
            print(f"skip  {tile_dir.name}  (missing he.png / ihc.png)")
            continue
        print(f"align {tile_dir.name} ...", end="", flush=True)
        try:
            result = register_tile(tile_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        (tile_dir / ALG_DIR).mkdir(exist_ok=True)
        out_file.write_text(json.dumps(result))
        print(f"  dx={result['dx']:.2f}  dy={result['dy']:.2f}")


def process_pairs(n: int, force: bool) -> None:
    if not DATA_ROOT.is_dir():
        sys.exit(f"Data root not found: {DATA_ROOT}")

    pair_dirs = sorted(
        (d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    queued = [d for d in pair_dirs if force or not pair_is_processed(d)]
    batch = queued[:n]

    if not batch:
        print("Nothing to process — all pairs already have results (use --force to rerun).")
        return

    total_done = total_skip = 0
    for pair_dir in batch:
        pair_id = pair_dir.name
        depth_dirs = sorted(
            (d for d in pair_dir.iterdir() if d.is_dir() and d.name.startswith("d")),
            key=lambda d: int(d.name[1:]),
        )
        for depth_dir in depth_dirs:
            level = int(depth_dir.name[1:])
            tile_count = sum(1 for d in depth_dir.iterdir() if d.is_dir())
            print(f"pair {pair_id}  level {level}  ({tile_count} tiles) ...", end="", flush=True)
            done, skip = process_depth(depth_dir, force)
            print(f"  {done} registered, {skip} skipped")
            total_done += done
            total_skip += skip

    print(f"\ntotal: {total_done} registered, {total_skip} skipped")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    argv = sys.argv[1:]
    force = "--force" in argv or "-f" in argv
    argv = [a for a in argv if a not in ("--force", "-f")]

    if "--pairs" in argv:
        idx = argv.index("--pairs")
        if idx + 1 >= len(argv):
            sys.exit("--pairs requires a number argument")
        n = int(argv[idx + 1])
        process_pairs(n, force=force)
        return

    if len(argv) < 2:
        sys.exit(
            "Usage:\n"
            "  python align.py <pair_id> <depth> [tile_id ...] [--force]\n"
            "  python align.py --pairs N [--force]"
        )
    pair_id  = int(argv[0])
    depth    = int(argv[1])
    tile_ids = argv[2:]
    process(pair_id, depth, tile_ids, force=force)


if __name__ == "__main__":
    main()
