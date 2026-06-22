"""
LNCC² metrics computation for HE/IHC tile pairs.

For each tile that already has an elastix/displacement.json, computes:
  lncc2       — LNCC² between normalised HE and IHC (no shift)
  lncc2_auto  — LNCC² after applying the auto displacement
  delta_px    — Euclidean length of the displacement vector in pixels
  factor_auto — lncc2_auto / lncc2

Results are written to data/cropped/<pair>/d<depth>/<tile>/metrics.json.

Usage — single depth:
    python metrics.py <pair_id> <depth> [--patch-size N] [--force]

Usage — batch (next N pairs that have alignments but no metrics):
    python metrics.py --pairs N [--patch-size N] [--force]

Default patch size: 50 px  (matches the canonical value used in the JS frontend).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import shift as ndimage_shift

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "cropped"
DISP_FILE = "elastix/displacement.json"
METRICS_FILE = "metrics.json"
DEFAULT_PATCH = 50


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_normalized_gray(path: Path) -> np.ndarray:
    """
    Load image, apply the same z-score normalization as the JS frontend
    (normalizeImageData: target mean=128, std=64), return float32 gray array.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR).astype(np.float64)
    gray_raw = img.mean(axis=2)
    mean = gray_raw.mean()
    std = gray_raw.std() or 1.0
    gray_norm = (gray_raw - mean) / std * 64.0 + 128.0
    return np.clip(gray_norm, 0, 255).astype(np.float32)


def shift_gray(gray: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift image by (dx, dy) with constant fill at 128 (neutral after normalization)."""
    return ndimage_shift(gray, shift=(dy, dx), order=1, mode="constant", cval=128.0).astype(np.float32)


# ── LNCC² (numpy SAT — mirrors computeLNCC in imageUtils.ts) ─────────────────

def _build_sat(arr: np.ndarray) -> np.ndarray:
    sat = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(arr.astype(np.float64), axis=0), axis=1)
    return sat


def compute_lncc2(g1: np.ndarray, g2: np.ndarray, patch_size: int) -> float:
    r = patch_size // 2
    area = float(patch_size * patch_size)
    h, w = g1.shape

    sat1   = _build_sat(g1)
    sat2   = _build_sat(g2)
    sat1sq = _build_sat(g1.astype(np.float64) ** 2)
    sat2sq = _build_sat(g2.astype(np.float64) ** 2)
    sat12  = _build_sat(g1.astype(np.float64) * g2.astype(np.float64))

    ys = np.arange(r, h - r)
    xs = np.arange(r, w - r)
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    def rs(sat: np.ndarray) -> np.ndarray:
        return (sat[Y + r + 1, X + r + 1]
                - sat[Y - r,     X + r + 1]
                - sat[Y + r + 1, X - r    ]
                + sat[Y - r,     X - r    ])

    s1   = rs(sat1)
    s2   = rs(sat2)
    s1sq = rs(sat1sq)
    s2sq = rs(sat2sq)
    s12  = rs(sat12)

    mu1  = s1 / area
    mu2  = s2 / area
    num  = s12 - area * mu1 * mu2
    den1 = np.maximum(0.0, s1sq - area * mu1 ** 2)
    den2 = np.maximum(0.0, s2sq - area * mu2 ** 2)
    den  = den1 * den2

    valid = den > 1e-6
    scores = np.where(valid, num ** 2 / np.where(valid, den, 1.0), 0.0)
    n = int(valid.sum())
    return float(scores.sum() / n) if n > 0 else 0.0


# ── Per-tile computation ──────────────────────────────────────────────────────

def compute_tile(tile_dir: Path, patch_size: int) -> dict[str, float]:
    disp = json.loads((tile_dir / DISP_FILE).read_text())
    dx, dy = float(disp["dx"]), float(disp["dy"])

    g1 = load_normalized_gray(tile_dir / "he.png")
    g2 = load_normalized_gray(tile_dir / "ihc.png")

    lncc2_base = compute_lncc2(g1, g2, patch_size)
    g2_shifted = shift_gray(g2, dx, dy)
    lncc2_auto = compute_lncc2(g1, g2_shifted, patch_size)
    delta_px   = math.sqrt(dx ** 2 + dy ** 2)
    factor     = lncc2_auto / lncc2_base if lncc2_base > 1e-9 else 0.0

    return {
        "lncc2": lncc2_base,
        "lncc2_auto": lncc2_auto,
        "delta_px": delta_px,
        "factor_auto": factor,
        "patch_size": patch_size,
    }


# ── Depth / pair processing ───────────────────────────────────────────────────

def process_depth(depth_dir: Path, patch_size: int, force: bool) -> tuple[int, int]:
    done = skipped = 0
    for tile_dir in sorted(d for d in depth_dir.iterdir() if d.is_dir()):
        disp_path   = tile_dir / DISP_FILE
        metric_path = tile_dir / METRICS_FILE
        if not disp_path.exists():
            skipped += 1
            continue
        if metric_path.exists() and not force:
            skipped += 1
            continue
        try:
            result = compute_tile(tile_dir, patch_size)
        except Exception as exc:
            print(f"    ERROR {tile_dir.name}: {exc}")
            skipped += 1
            continue
        metric_path.write_text(json.dumps(result))
        done += 1
    return done, skipped


def pair_has_metrics(pair_dir: Path) -> bool:
    return any(pair_dir.rglob(METRICS_FILE))


def process(pair_id: int, depth: int, patch_size: int, force: bool) -> None:
    depth_dir = DATA_ROOT / str(pair_id) / f"d{depth}"
    if not depth_dir.is_dir():
        sys.exit(f"Directory not found: {depth_dir}")
    done, skip = process_depth(depth_dir, patch_size, force)
    print(f"pair {pair_id}  level {depth}  done={done}  skipped={skip}")


def process_pairs(n: int, patch_size: int, force: bool) -> None:
    if not DATA_ROOT.is_dir():
        sys.exit(f"Data root not found: {DATA_ROOT}")

    pair_dirs = sorted(
        (d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    queued = [d for d in pair_dirs if force or not pair_has_metrics(d)]
    batch  = queued[:n]

    if not batch:
        print("Nothing to process — all pairs already have metrics (use --force to rerun).")
        return

    total_done = total_skip = 0
    for pair_dir in batch:
        pair_id   = pair_dir.name
        depth_dirs = sorted(
            (d for d in pair_dir.iterdir() if d.is_dir() and d.name.startswith("d")),
            key=lambda d: int(d.name[1:]),
        )
        for depth_dir in depth_dirs:
            level = int(depth_dir.name[1:])
            tile_count = sum(1 for d in depth_dir.iterdir() if d.is_dir())
            print(f"pair {pair_id}  level {level}  ({tile_count} tiles) ...", end="", flush=True)
            done, skip = process_depth(depth_dir, patch_size, force)
            print(f"  {done} computed, {skip} skipped")
            total_done += done
            total_skip += skip

    print(f"\ntotal: {total_done} computed, {total_skip} skipped")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    argv  = sys.argv[1:]
    force = "--force" in argv or "-f" in argv
    argv  = [a for a in argv if a not in ("--force", "-f")]

    patch_size = DEFAULT_PATCH
    if "--patch-size" in argv:
        idx = argv.index("--patch-size")
        if idx + 1 >= len(argv):
            sys.exit("--patch-size requires a number argument")
        patch_size = int(argv[idx + 1])
        argv = argv[:idx] + argv[idx + 2:]

    if "--pairs" in argv:
        idx = argv.index("--pairs")
        if idx + 1 >= len(argv):
            sys.exit("--pairs requires a number argument")
        n = int(argv[idx + 1])
        process_pairs(n, patch_size=patch_size, force=force)
        return

    if len(argv) < 2:
        sys.exit(
            "Usage:\n"
            "  python metrics.py <pair_id> <depth> [--patch-size N] [--force]\n"
            "  python metrics.py --pairs N [--patch-size N] [--force]"
        )
    process(int(argv[0]), int(argv[1]), patch_size=patch_size, force=force)


if __name__ == "__main__":
    main()
