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

EXCLUDE_BACKGROUND = True
CROPPED_DIR = conf.PROJECT_ROOT / "data" / "cropped"
ANNOTATION_PATH = conf.ANNOTATION_PATH
MAX_CROP_DEPTH = conf.MAX_CROP_DEPTH
PAIR_IDS = None
OVERWRITE = False
LOG_EVERY = 100

CNN_INPUT_HEIGHT = conf.CNN_INPUT_HEIGHT
CNN_INPUT_WIDTH = conf.CNN_INPUT_WIDTH


def paired_tile_paths(pair_id, crop_depth, x_idx, y_idx):
    leaf = CROPPED_DIR / str(pair_id) / f"d{crop_depth}" / f"{x_idx}_{y_idx}"
    return leaf / "he.png", leaf / "ihc.png"


def background_tile_paths(pair_id, crop_depth, x_idx, y_idx):
    stem = f"{pair_id}_d{crop_depth}_{x_idx}_{y_idx}"
    folder = CROPPED_DIR / "background"
    return folder / f"{stem}_he.png", folder / f"{stem}_ihc.png"


def test_class_tile_paths(test_run_class, pair_id, crop_depth, x_idx, y_idx, is_background):
    sub = "background" if is_background else "valid"
    folder = CROPPED_DIR / "test" / test_run_class / sub
    stem = f"{pair_id}_d{crop_depth}_{x_idx}_{y_idx}"
    return folder / f"{stem}_he.png", folder / f"{stem}_ihc.png"


def output_paths(job):
    pair_id = job["pair_id"]
    crop_depth = job["crop_depth"]
    x_idx = job["x_idx"]
    y_idx = job["y_idx"]
    is_background = bool(job.get("binary_mask_excluded", False))
    test_run_class = job.get("test_run_class")

    if test_run_class:
        return test_class_tile_paths(
            test_run_class, pair_id, crop_depth, x_idx, y_idx, is_background
        )
    if is_background and EXCLUDE_BACKGROUND:
        return background_tile_paths(pair_id, crop_depth, x_idx, y_idx)
    return paired_tile_paths(pair_id, crop_depth, x_idx, y_idx)


def load_page(path, pyramid_page_idx, page_cache):
    path = conf.resolve(path)
    key = (str(path), int(pyramid_page_idx))
    if key not in page_cache:
        with tifffile.TiffFile(path) as slide:
            page_cache[key] = slide.pages[pyramid_page_idx].asarray()
    return page_cache[key]


def crop_tile(img, x_idx, y_idx, grid):
    h, w = img.shape[:2]
    tile_w = w // grid
    tile_h = h // grid
    x0 = x_idx * tile_w
    y0 = y_idx * tile_h
    x1 = w if x_idx == grid - 1 else x0 + tile_w
    y1 = h if y_idx == grid - 1 else y0 + tile_h
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


def save_side(job, side, dest, page_cache):
    path = conf.job_image_path(job, side)
    page = load_page(path, job["pyramid_page_idx"], page_cache)
    tile = crop_tile(page, job["x_idx"], job["y_idx"], job["grid"])
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


def main():
    print(f"System          : {conf.SYSTEM_PREFIX!r}")
    print(f"Annotation path : {ANNOTATION_PATH}")
    print(f"Output dir      : {CROPPED_DIR}")
    print(f"Max crop depth  : {MAX_CROP_DEPTH}")
    print(f"Exclude bg      : {EXCLUDE_BACKGROUND}")
    print(f"Overwrite       : {OVERWRITE}")
    print()

    if not ANNOTATION_PATH.exists():
        print(f"[ERROR] Annotation file not found: {ANNOTATION_PATH}")
        sys.exit(1)

    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    jobs = filter_jobs(jobs)
    is_test_json = any(job.get("test_run_class") for job in jobs)
    print(f"Mode            : {'test (test_run_class)' if is_test_json else 'production'}")
    print(f"Jobs to process : {len(jobs)}")

    page_cache = {}
    counts = {"passed": 0, "background": 0, "skipped": 0, "written": 0}
    class_counts = defaultdict(lambda: {"valid": 0, "background": 0})

    for idx, job in enumerate(jobs, start=1):
        he_path, ihc_path = output_paths(job)

        if should_skip(he_path, ihc_path):
            counts["skipped"] += 1
            continue

        save_side(job, "fixed", he_path, page_cache)
        save_side(job, "moving", ihc_path, page_cache)
        counts["written"] += 1

        is_bg = job.get("binary_mask_excluded", False) and EXCLUDE_BACKGROUND
        if is_bg:
            counts["background"] += 1
        else:
            counts["passed"] += 1

        test_run_class = job.get("test_run_class")
        if test_run_class:
            bucket = "background" if is_bg else "valid"
            class_counts[test_run_class][bucket] += 1

        if idx % LOG_EVERY == 0 or idx == len(jobs):
            print(
                f"  [{idx}/{len(jobs)}] written={counts['written']} "
                f"skipped={counts['skipped']}"
            )

    print()
    if class_counts:
        print("Per config:")
        for class_name in sorted(class_counts):
            valid = class_counts[class_name]["valid"]
            background = class_counts[class_name]["background"]
            print(f"  {class_name}: valid={valid} background={background}")
        print()
    print(f"Valid jobs written     : {counts['passed']}")
    print(f"Background jobs written: {counts['background']}")
    print(f"Skipped (existing)     : {counts['skipped']}")
    print(f"PNG files on disk      : {counts['written'] * 2}")


if __name__ == "__main__":
    main()
