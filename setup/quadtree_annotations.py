"""
Builds the quadtree tile index JSON (replaces quadtree_annotations.ipynb).

For each image pair, selects the coarsest TIFF pyramid page that still covers
the CNN input dimensions at each quadtree depth, then records one annotation
entry per (pair, depth, x, y) tile.

Usage:
  python setup/quadtree_annotations.py [--pair-ids N ...] [--smooth] [--force]

Flags:
  --pair-ids N ...  Process only these pair IDs (default: all).
  --smooth          Write output to data/smooth/ instead of data/.
                    Content is identical; the path signals downstream scripts
                    (preprocess_tiles.py --smooth) to use smooth-displaced tiles.
  --force           Overwrite existing annotation file.
"""

import argparse
import json
import sys
from pathlib import Path

import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import conf
from pair_mask import (
    PRODUCTION_MIN_INSIDE_FRACTION,
    is_tile_excluded_by_polygons,
    load_pair_mask,
    mask_json_path,
    scaled_polygons_for_page,
)

LOG_EVERY        = 300
IMAGE_DIR        = conf.IMAGE_DIR
LABELS_PATH      = conf.LABELS_PATH
WSI_PAGES        = conf.WSI_PAGES
CNN_INPUT_HEIGHT = conf.CNN_INPUT_HEIGHT
CNN_INPUT_WIDTH  = conf.CNN_INPUT_WIDTH
MAX_CROP_DEPTH   = conf.MAX_CROP_DEPTH


def _matrix_dict_to_list(m):
    return [
        [m["t_00"], m["t_01"], m["t_02"]],
        [m["t_10"], m["t_11"], m["t_12"]],
        [m["t_20"], m["t_21"], m["t_22"]],
    ]


def _load_pairs(pair_ids_filter=None):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    pairs = []
    for pair_id, item in enumerate(labels):
        if pair_ids_filter is not None and pair_id not in pair_ids_filter:
            continue
        source_id = item["source_image_id"]
        target_id = item["target_image_id"]
        source_path = IMAGE_DIR / f"{source_id}.data"
        target_path = IMAGE_DIR / f"{target_id}.data"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        pairs.append({
            "pair_id":              pair_id,
            "moving_path":          conf.image_relpath(source_id),
            "fixed_path":           conf.image_relpath(target_id),
            "source_image_id":      source_id,
            "target_image_id":      target_id,
            "registration_error":   item["registration_error"],
            "transformation_matrix": _matrix_dict_to_list(item["transformation_matrix"]),
        })
    return pairs


def _choose_pyramid_page(fixed_path, moving_path, crop_depth):
    fixed_path  = conf.resolve(fixed_path)
    moving_path = conf.resolve(moving_path)
    grid = 2 ** crop_depth
    with (
        tifffile.TiffFile(fixed_path)  as fixed_slide,
        tifffile.TiffFile(moving_path) as moving_slide,
    ):
        candidates = []
        for page_idx in WSI_PAGES:
            fh, fw = fixed_slide.pages[page_idx].shape[:2]
            mh, mw = moving_slide.pages[page_idx].shape[:2]
            min_tile_h = min(fh // grid, mh // grid)
            min_tile_w = min(fw // grid, mw // grid)
            if min_tile_h >= CNN_INPUT_HEIGHT and min_tile_w >= CNN_INPUT_WIDTH:
                candidates.append((page_idx, min_tile_h, min_tile_w))
    return candidates[-1] if candidates else None


def _page_shape(path, pyramid_page_idx, shape_cache):
    path = conf.resolve(path)
    key  = (str(path), pyramid_page_idx)
    if key not in shape_cache:
        with tifffile.TiffFile(path) as slide:
            shape_cache[key] = slide.pages[pyramid_page_idx].shape[:2]
    return shape_cache[key]


def _he_tile_excluded(pair, job, shape_cache, polygon_cache):
    page_h, page_w = _page_shape(pair["fixed_path"], job["pyramid_page_idx"], shape_cache)
    poly_key = (pair["pair_id"], page_w, page_h)
    if poly_key not in polygon_cache:
        polygons = scaled_polygons_for_page(pair["pair_id"], page_w, page_h)
        if polygons is None:
            raise FileNotFoundError(
                f"No HE mask for pair {pair['pair_id']}: {mask_json_path(pair['pair_id'])}"
            )
        polygon_cache[poly_key] = polygons
    return is_tile_excluded_by_polygons(
        polygon_cache[poly_key],
        job["grid"], job["x_idx"], job["y_idx"],
        page_h, page_w,
        PRODUCTION_MIN_INSIDE_FRACTION,
    )


def _build_index(pairs, save_path):
    tile_jobs    = []
    shape_cache  = {}
    polygon_cache = {}

    try:
        for pair in pairs:
            print(
                f"pair_id {pair['pair_id']}: "
                f"HE {pair['target_image_id']} / IHC {pair['source_image_id']}"
            )
            for crop_depth in range(MAX_CROP_DEPTH + 1):
                chosen = _choose_pyramid_page(
                    pair["fixed_path"], pair["moving_path"], crop_depth
                )
                if chosen is None:
                    continue
                pyramid_page_idx, tile_h, tile_w = chosen
                grid = 2 ** crop_depth
                print(
                    f"  d{crop_depth}: grid {grid}x{grid}, "
                    f"page {pyramid_page_idx}, tile {tile_w}x{tile_h}"
                )
                for y_idx in range(grid):
                    for x_idx in range(grid):
                        job = {
                            "pair_id":              pair["pair_id"],
                            "fixed_path":           pair["fixed_path"],
                            "moving_path":          pair["moving_path"],
                            "source_image_id":      pair["source_image_id"],
                            "target_image_id":      pair["target_image_id"],
                            "crop_depth":           crop_depth,
                            "grid":                 grid,
                            "x_idx":                x_idx,
                            "y_idx":                y_idx,
                            "pyramid_page_idx":     pyramid_page_idx,
                            "tile_h":               tile_h,
                            "tile_w":               tile_w,
                            "cnn_input_height":     CNN_INPUT_HEIGHT,
                            "cnn_input_width":      CNN_INPUT_WIDTH,
                            "registration_error":   pair["registration_error"],
                            "transformation_matrix": pair["transformation_matrix"],
                        }
                        job["binary_mask_excluded"] = _he_tile_excluded(
                            pair, job, shape_cache, polygon_cache
                        )
                        tile_jobs.append(job)
                        if LOG_EVERY and len(tile_jobs) % LOG_EVERY == 0:
                            print(
                                f"    [{len(tile_jobs)} tiles] "
                                f"pair_id={pair['pair_id']} d{crop_depth} ({x_idx},{y_idx})"
                            )
            print(f"  pair_id {pair['pair_id']} done — {len(tile_jobs)} tiles total so far")

    except KeyboardInterrupt:
        if tile_jobs:
            _save_json(tile_jobs, save_path)
            print(
                f"\nKeyboardInterrupt — partial index ({len(tile_jobs)} jobs) "
                f"saved to {save_path}"
            )
        raise SystemExit(0)

    return tile_jobs


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-ids", type=int, nargs="+",
                        help="Process only these pair IDs")
    parser.add_argument("--smooth",   action="store_true",
                        help="Write output to data/smooth/ instead of data/")
    parser.add_argument("--force",    action="store_true",
                        help="Overwrite existing annotation file")
    args = parser.parse_args()

    if args.smooth:
        out_path = conf.PROJECT_ROOT / "data" / "smooth" / Path(conf.ANNOTATION_PATH).name
    else:
        out_path = conf.ANNOTATION_PATH

    if out_path.exists() and not args.force:
        print(f"Annotation file already exists: {out_path}")
        print("Use --force to overwrite.")
        sys.exit(0)

    pair_ids_filter = set(args.pair_ids) if args.pair_ids else None
    pairs = _load_pairs(pair_ids_filter)

    print(f"System   : {conf.SYSTEM_PREFIX!r}")
    print(f"Labels   : {LABELS_PATH}")
    print(f"Output   : {out_path}")
    print(f"Smooth   : {args.smooth}")
    print(f"Pairs    : {len(pairs)}")
    print()

    tile_jobs = _build_index(pairs, save_path=out_path)

    excluded = sum(1 for j in tile_jobs if j["binary_mask_excluded"])
    _save_json(tile_jobs, out_path)

    print()
    print(f"pairs     : {len(pairs)}")
    print(f"tile jobs : {len(tile_jobs)}")
    print(f"excluded  : {excluded}/{len(tile_jobs)} background")
    print(f"saved to  : {out_path}")


if __name__ == "__main__":
    main()
