"""
Replace multi-page WSI pyramids with single-page L6 RGB BigTIFFs.

Pair-scoped: for each HE/IHC pair, convert one slide in a subprocess (pyvips
stream), replace+delete the original, then the other slide, then the next pair.

RAM: never NumPy-decode a full pyramid page. Child process exits after each
slide so macOS can reclaim buffers.

Usage:
  python setup/downsample_images_l6.py --pairs 0 --commit
  python setup/downsample_images_l6.py --commit
  python setup/downsample_images_l6.py --remap-annotations-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import conf

NEED_H = 64 * conf.CNN_INPUT_HEIGHT
NEED_W = 64 * conf.CNN_INPUT_WIDTH
IMAGE_DIR = conf.IMAGE_DIR
TMP_SUFFIX = ".l6tmp"
BAK_SUFFIX = ".bak"
VIPS_CACHE_BYTES = 256 * 1024 * 1024


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _page_shape(path: Path) -> tuple[int, int, int]:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        n = len(tif.pages)
        h, w = tif.pages[0].shape[:2]
        return n, h, w


def _already_l6(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        n, h, w = _page_shape(path)
        if n == 1 and h >= NEED_H and w >= NEED_W:
            return True, (h, w)
        return False, (h, w)
    except Exception:
        return False, None


def _pick_openslide_level(path: Path) -> int:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        chosen = 0
        for level in range(slide.level_count):
            lw, lh = slide.level_dimensions[level]
            if lh >= NEED_H and lw >= NEED_W:
                chosen = level
            else:
                break
        return chosen
    finally:
        slide.close()


def worker_convert(src: Path, dst: Path) -> None:
    import pyvips

    pyvips.cache_set_max(0)
    pyvips.cache_set_max_mem(VIPS_CACHE_BYTES)
    pyvips.cache_set_max_files(8)

    level = _pick_openslide_level(src)
    print(f"  worker level={level} rss_mb={_rss_mb():.0f}", flush=True)

    img = pyvips.Image.openslideload(str(src), level=level, access="sequential")
    if img.bands >= 4:
        img = img.extract_band(0, n=3)
    elif img.bands == 1:
        img = img.bandjoin([img, img])

    scale = max(NEED_H / img.height, NEED_W / img.width)
    if abs(scale - 1.0) > 1e-6:
        img = img.resize(scale, kernel="lanczos3")

    if img.height < NEED_H or img.width < NEED_W:
        raise RuntimeError(
            f"after resize got {img.height}x{img.width}, need >={NEED_H}x{NEED_W}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    img.tiffsave(
        str(dst),
        compression="deflate",
        tile=True,
        tile_width=512,
        tile_height=512,
        bigtiff=True,
        pyramid=False,
    )
    print(
        f"  worker wrote {img.height}x{img.width} -> {dst} "
        f"({dst.stat().st_size / 1e9:.2f} GB) rss_mb={_rss_mb():.0f}",
        flush=True,
    )


def _run_worker(src: Path, dst: Path) -> None:
    env = os.environ.copy()
    env["VIPS_CONCURRENCY"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(src),
        str(dst),
    ]
    print(f"  spawn worker rss_parent_mb={_rss_mb():.0f}", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed exit={proc.returncode} for {src.name}")


def _verify_meta(path: Path) -> tuple[int, int]:
    n, h, w = _page_shape(path)
    if n < 1:
        raise RuntimeError(f"{path}: no pages")
    if h < NEED_H or w < NEED_W:
        raise RuntimeError(f"{path}: shape {h}x{w} below {NEED_H}x{NEED_W}")
    return h, w


def convert_slide(image_id: int, commit: bool) -> dict:
    path = IMAGE_DIR / f"{image_id}.data"
    if not path.is_file():
        raise FileNotFoundError(path)

    old_bytes = path.stat().st_size
    ok, shape = _already_l6(path)
    if ok and shape is not None:
        print(f"SKIP {image_id}: already L6 {shape[0]}x{shape[1]}", flush=True)
        return {
            "image_id": image_id,
            "status": "skipped",
            "old_bytes": old_bytes,
            "new_bytes": old_bytes,
            "shape": shape,
        }

    print(f"START {image_id}: {old_bytes / 1e9:.2f} GB", flush=True)
    tmp = Path(str(path) + TMP_SUFFIX)
    bak = Path(str(path) + BAK_SUFFIX)
    if tmp.exists():
        tmp.unlink()

    _run_worker(path, tmp)
    h, w = _verify_meta(tmp)
    new_bytes = tmp.stat().st_size

    if not commit:
        tmp.unlink(missing_ok=True)
        print(f"DRY {image_id}: would be {h}x{w} ({new_bytes / 1e9:.2f} GB)", flush=True)
        return {
            "image_id": image_id,
            "status": "dry_run",
            "old_bytes": old_bytes,
            "new_bytes": new_bytes,
            "shape": (h, w),
        }

    if bak.exists():
        bak.unlink()
    path.rename(bak)
    tmp.rename(path)
    try:
        _verify_meta(path)
    except Exception:
        if path.exists():
            path.unlink()
        bak.rename(path)
        raise
    bak.unlink()
    gc.collect()
    print(
        f"DONE {image_id}: {old_bytes / 1e9:.2f} -> {new_bytes / 1e9:.2f} GB "
        f"rss_mb={_rss_mb():.0f}",
        flush=True,
    )
    return {
        "image_id": image_id,
        "status": "replaced",
        "old_bytes": old_bytes,
        "new_bytes": new_bytes,
        "shape": (h, w),
    }


def _pairs(explicit: list[int] | None) -> list[tuple[int, int, int]]:
    labels = json.loads(conf.LABELS_PATH.read_text())
    out = []
    for i, item in enumerate(labels):
        if explicit is not None and i not in explicit:
            continue
        he = int(item["target_image_id"])
        ihc = int(item["source_image_id"])
        out.append((i, he, ihc))
    return out


def convert_pair(pair_id: int, he_id: int, ihc_id: int, commit: bool) -> list[dict]:
    print(f"\n=== PAIR {pair_id}: HE={he_id} IHC={ihc_id} ===", flush=True)
    results = []
    results.append(convert_slide(he_id, commit=commit))
    gc.collect()
    results.append(convert_slide(ihc_id, commit=commit))
    gc.collect()
    return results


def remap_annotations(ann_path: Path) -> int:
    import tifffile

    if not ann_path.is_file():
        print(f"no annotations at {ann_path}", flush=True)
        return 0

    jobs = json.loads(ann_path.read_text())
    shape_cache: dict[int, tuple[int, int]] = {}

    def page_shape(image_id: int) -> tuple[int, int]:
        if image_id not in shape_cache:
            path = IMAGE_DIR / f"{image_id}.data"
            with tifffile.TiffFile(str(path)) as tif:
                h, w = tif.pages[0].shape[:2]
            shape_cache[image_id] = (h, w)
        return shape_cache[image_id]

    n = 0
    for job in jobs:
        tid = int(job["target_image_id"])
        h, w = page_shape(tid)
        grid = int(job["grid"])
        job["pyramid_page_idx"] = 0
        job["tile_h"] = h // grid
        job["tile_w"] = w // grid
        n += 1

    ann_path.write_text(json.dumps(jobs))
    print(f"remapped {n} jobs in {ann_path}", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=2, metavar=("SRC", "DST"), help=argparse.SUPPRESS)
    ap.add_argument("--pairs", nargs="+", type=int, help="pair indices to process")
    ap.add_argument("--commit", action="store_true", help="replace files (otherwise dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="alias for non-commit mode")
    ap.add_argument("--remap-annotations-only", action="store_true")
    ap.add_argument("--skip-remap", action="store_true")
    args = ap.parse_args()

    if args.worker:
        src, dst = Path(args.worker[0]), Path(args.worker[1])
        worker_convert(src, dst)
        return

    if args.remap_annotations_only:
        remap_annotations(conf.PROJECT_ROOT / "data" / "macos_quadtree_annotations.json")
        linux = conf.PROJECT_ROOT / "data" / "linux_quadtree_annotations.json"
        if linux.is_file():
            remap_annotations(linux)
        return

    commit = bool(args.commit) and not args.dry_run
    pairs = _pairs(args.pairs)
    print(
        f"L6 target {NEED_H}x{NEED_W}; pairs={len(pairs)}; commit={commit}",
        flush=True,
    )

    results: list[dict] = []
    for pair_id, he_id, ihc_id in pairs:
        try:
            results.extend(convert_pair(pair_id, he_id, ihc_id, commit=commit))
        except Exception as e:
            print(f"FAIL pair {pair_id}: {e}", flush=True)
            if commit:
                raise

    old_total = sum(r.get("old_bytes", 0) for r in results)
    new_total = sum(r.get("new_bytes", 0) for r in results)
    print(
        json.dumps(
            {
                "n_slides": len(results),
                "old_gb": round(old_total / 1e9, 2),
                "new_gb": round(new_total / 1e9, 2),
                "freed_gb": round((old_total - new_total) / 1e9, 2),
                "statuses": {
                    s: sum(1 for r in results if r.get("status") == s)
                    for s in sorted({r.get("status") for r in results})
                },
            },
            indent=2,
        ),
        flush=True,
    )

    if commit and not args.skip_remap:
        remap_annotations(conf.PROJECT_ROOT / "data" / "macos_quadtree_annotations.json")
        linux = conf.PROJECT_ROOT / "data" / "linux_quadtree_annotations.json"
        if linux.is_file():
            remap_annotations(linux)


if __name__ == "__main__":
    main()
