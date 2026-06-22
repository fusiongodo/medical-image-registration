import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf

TEST_LAYOUT = False
EXCLUDE_BACKGROUND = True
CROPPED_DIR = conf.PROJECT_ROOT / "data" / "cropped"
SMOOTH_DIR  = conf.PROJECT_ROOT / "data" / "smooth"
ANNOTATION_PATH = conf.ANNOTATION_PATH
MAX_CROP_DEPTH = conf.MAX_CROP_DEPTH
PAIR_IDS = None
OVERWRITE = False
LOG_EVERY = 100

CNN_INPUT_HEIGHT = conf.CNN_INPUT_HEIGHT
CNN_INPUT_WIDTH  = conf.CNN_INPUT_WIDTH


def paired_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx):
    leaf = out_dir / str(pair_id) / f"d{crop_depth}" / f"{x_idx}_{y_idx}"
    return leaf / "he.png", leaf / "ihc.png"


def background_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx):
    stem = f"{pair_id}_d{crop_depth}_{x_idx}_{y_idx}"
    folder = out_dir / "background"
    return folder / f"{stem}_he.png", folder / f"{stem}_ihc.png"


def test_valid_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx):
    folder = out_dir / "test" / f"d{crop_depth}"
    stem = f"{pair_id}_{x_idx}_{y_idx}"
    return folder / f"{stem}_he.png", folder / f"{stem}_ihc.png"


def test_background_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx):
    folder = out_dir / "test" / "background"
    stem = f"{pair_id}_d{crop_depth}_{x_idx}_{y_idx}"
    return folder / f"{stem}_he.png", folder / f"{stem}_ihc.png"


def output_paths(job, out_dir):
    pair_id    = job["pair_id"]
    crop_depth = job["crop_depth"]
    x_idx      = job["x_idx"]
    y_idx      = job["y_idx"]
    is_background = bool(job.get("binary_mask_excluded", False))

    if TEST_LAYOUT:
        if is_background and EXCLUDE_BACKGROUND:
            return test_background_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx)
        return test_valid_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx)
    if is_background and EXCLUDE_BACKGROUND:
        return background_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx)
    return paired_tile_paths(out_dir, pair_id, crop_depth, x_idx, y_idx)


_BAD_PAGES = set()


def load_page(path, pyramid_page_idx, page_cache):
    path = conf.resolve(path)
    key = (str(path), int(pyramid_page_idx))
    if key in _BAD_PAGES:
        raise RuntimeError(f"bad page (cached): {path} page {pyramid_page_idx}")
    if key not in page_cache:
        with tifffile.TiffFile(path) as slide:
            try:
                page_cache[key] = slide.pages[pyramid_page_idx].asarray()
            except Exception as exc:
                _BAD_PAGES.add(key)
                raise RuntimeError(
                    f"Failed to decode {path} page {pyramid_page_idx}: {exc}"
                ) from exc
    return page_cache[key]


def crop_tile(img, x_idx, y_idx, grid, dx_wsi=0, dy_wsi=0):
    """
    Crop one tile.  For smooth IHC extraction dx_wsi/dy_wsi are subtracted:
    if align.py says "shift IHC right by dx to align", the crop window must
    move left by dx_wsi so the resulting tile is already in registration.
    """
    h, w = img.shape[:2]
    tile_w = w // grid
    tile_h = h // grid
    x0 = int(round(x_idx * tile_w - dx_wsi))
    y0 = int(round(y_idx * tile_h - dy_wsi))
    x0 = max(0, min(x0, w - tile_w))
    y0 = max(0, min(y0, h - tile_h))
    x1 = x0 + tile_w
    y1 = y0 + tile_h
    return img[y0:y1, x0:x1]


def tile_to_gray_png_array(tile):
    if tile.dtype != np.uint8:
        tile = tile.astype(np.uint8)
    image = Image.fromarray(tile).convert("L")
    image = image.resize(
        (CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT),
        resample=Image.BILINEAR,
    )
    return image


def save_side(job, side, dest, page_cache, dx_wsi=0, dy_wsi=0):
    path = conf.job_image_path(job, side)
    page = load_page(path, job["pyramid_page_idx"], page_cache)
    tile = crop_tile(page, job["x_idx"], job["y_idx"], job["grid"], dx_wsi, dy_wsi)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tile_to_gray_png_array(tile).save(dest, format="PNG")


def should_skip(he_path, ihc_path):
    if OVERWRITE:
        return False
    return he_path.exists() and ihc_path.exists()


def filter_jobs(jobs):
    filtered = []
    for job in jobs:
        if job["crop_depth"] > MAX_CROP_DEPTH:
            continue
        if PAIR_IDS is not None and job["pair_id"] not in PAIR_IDS:
            continue
        filtered.append(job)
    return filtered


_smooth_cache: dict[int, dict] = {}


def _smooth_offset(pair_id, crop_depth, x_idx, y_idx, page_shape):
    """
    Return (dx_wsi, dy_wsi) in WSI-page pixels for smooth IHC extraction.
    page_shape = (h, w) of the loaded WSI page.
    """
    if pair_id not in _smooth_cache:
        sf_path = SMOOTH_DIR / f"{pair_id}_smooth_field.json"
        if not sf_path.exists():
            _smooth_cache[pair_id] = {}
        else:
            _smooth_cache[pair_id] = json.loads(sf_path.read_text())
    sf = _smooth_cache[pair_id]
    depths = sf.get("depths", {})
    tile_entry = depths.get(str(crop_depth), {}).get(f"{x_idx}_{y_idx}")
    if tile_entry is None:
        return 0.0, 0.0

    dx_tile = tile_entry["dx"]
    dy_tile = tile_entry["dy"]

    page_h, page_w = page_shape
    grid = 2 ** crop_depth
    tile_w = page_w // grid
    tile_h = page_h // grid
    dx_wsi = dx_tile * tile_w / CNN_INPUT_WIDTH
    dy_wsi = dy_tile * tile_h / CNN_INPUT_HEIGHT
    return dx_wsi, dy_wsi


def main():
    parser = argparse.ArgumentParser(description="Extract paired HE/IHC tiles from WSI.")
    parser.add_argument("--smooth",       action="store_true",
                        help="Apply smooth translation field to IHC crops; "
                             "output goes to data/cropped_smooth/")
    parser.add_argument("--smooth-dir",   type=Path, default=None,
                        help="Override directory containing smooth_field.json files "
                             "(default: data/smooth)")
    parser.add_argument("--annotations",  type=Path, default=None,
                        help="Override annotation JSON path")
    parser.add_argument("--pair-ids",     type=int, nargs="+",
                        help="Process only these pair IDs")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract tiles that already exist")
    args = parser.parse_args()

    global OVERWRITE, PAIR_IDS, SMOOTH_DIR
    OVERWRITE = args.overwrite
    if args.pair_ids:
        PAIR_IDS = set(args.pair_ids)
    if args.smooth_dir:
        SMOOTH_DIR = args.smooth_dir

    annotation_path = args.annotations or ANNOTATION_PATH
    out_dir = conf.PROJECT_ROOT / "data" / ("cropped_smooth" if args.smooth else "cropped")

    print(f"System          : {conf.SYSTEM_PREFIX!r}")
    print(f"Annotation path : {annotation_path}")
    print(f"Output dir      : {out_dir}")
    print(f"Smooth IHC      : {args.smooth}")
    print(f"Layout          : {'test (flat by depth)' if TEST_LAYOUT else 'production'}")
    print(f"Max crop depth  : {MAX_CROP_DEPTH}")
    print(f"Exclude bg      : {EXCLUDE_BACKGROUND}")
    print(f"Overwrite       : {OVERWRITE}")
    print()

    if not annotation_path.exists():
        print(f"[ERROR] Annotation file not found: {annotation_path}")
        sys.exit(1)

    with open(annotation_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    jobs = filter_jobs(jobs)
    print(f"Jobs to process : {len(jobs)}")

    page_cache = {}
    counts = {"passed": 0, "background": 0, "skipped": 0, "written": 0}
    depth_counts = defaultdict(lambda: {"valid": 0, "background": 0})
    current_pair_id = None

    for idx, job in enumerate(jobs, start=1):
        if job["pair_id"] != current_pair_id:
            page_cache.clear()
            current_pair_id = job["pair_id"]

        he_path, ihc_path = output_paths(job, out_dir)

        if should_skip(he_path, ihc_path):
            counts["skipped"] += 1
            continue

        try:
            save_side(job, "fixed", he_path, page_cache)

            if args.smooth:
                ihc_page = load_page(
                    conf.job_image_path(job, "moving"),
                    job["pyramid_page_idx"],
                    page_cache,
                )
                dx_wsi, dy_wsi = _smooth_offset(
                    job["pair_id"], job["crop_depth"],
                    job["x_idx"], job["y_idx"],
                    ihc_page.shape[:2],
                )
                save_side(job, "moving", ihc_path, page_cache, dx_wsi, dy_wsi)
            else:
                save_side(job, "moving", ihc_path, page_cache)

        except RuntimeError as exc:
            msg = str(exc)
            if "bad page (cached)" not in msg:
                print(f"\n[WARN] {msg} — all tiles on this page will be skipped")
            counts["skipped"] += 1
            continue
        counts["written"] += 1

        is_bg = job.get("binary_mask_excluded", False) and EXCLUDE_BACKGROUND
        if is_bg:
            counts["background"] += 1
        else:
            counts["passed"] += 1

        if TEST_LAYOUT:
            bucket = "background" if is_bg else "valid"
            depth_counts[job["crop_depth"]][bucket] += 1

        if idx % LOG_EVERY == 0 or idx == len(jobs):
            print(
                f"  [{idx}/{len(jobs)}] written={counts['written']} "
                f"skipped={counts['skipped']}"
            )

    print()
    if depth_counts:
        print("Per depth:")
        for depth in sorted(depth_counts):
            valid = depth_counts[depth]["valid"]
            background = depth_counts[depth]["background"]
            print(f"  d{depth}: valid={valid} background={background}")
        print()
    print(f"Valid jobs written     : {counts['passed']}")
    print(f"Background jobs written: {counts['background']}")
    print(f"Skipped (existing)     : {counts['skipped']}")
    print(f"PNG files on disk      : {counts['written'] * 2}")


if __name__ == "__main__":
    main()
