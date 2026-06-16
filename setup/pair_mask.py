import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

import conf

MASK_DIR = conf.PROJECT_ROOT / "data" / "masks"

PRODUCTION_MIN_INSIDE_FRACTION = 0.33

TEST_RUN_AREA_THRESHOLDS = {
    "cfg_0": 0.40,
    "cfg_1": 0.35,
    "cfg_2": 0.33,
    "cfg_3": 0.28,
    "cfg_4": 0.20,
}


def mask_json_path(pair_id):
    return MASK_DIR / f"{pair_id}_he.json"


def mask_preview_path(pair_id):
    return MASK_DIR / f"{pair_id}_he_preview.png"


def scaled_polygon(polygon, src_width, src_height, dst_width, dst_height):
    if src_width == dst_width and src_height == dst_height:
        return polygon
    sx = dst_width / src_width
    sy = dst_height / src_height
    return [[x * sx, y * sy] for x, y in polygon]


def rasterize_polygon(polygon, width, height):
    if len(polygon) < 3:
        return np.zeros((height, width), dtype=bool)
    path = MplPath(polygon)
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.column_stack((xs.ravel(), ys.ravel()))
    return path.contains_points(points).reshape(height, width)


def tile_bbox(grid, x_idx, y_idx, page_height, page_width):
    tile_w = page_width // grid
    tile_h = page_height // grid
    x0 = x_idx * tile_w
    y0 = y_idx * tile_h
    x1 = page_width if x_idx == grid - 1 else x0 + tile_w
    y1 = page_height if y_idx == grid - 1 else y0 + tile_h
    return x0, y0, x1, y1


def tile_inside_fraction(mask, grid, x_idx, y_idx):
    page_height, page_width = mask.shape
    x0, y0, x1, y1 = tile_bbox(grid, x_idx, y_idx, page_height, page_width)
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def is_tile_excluded_by_mask(mask, grid, x_idx, y_idx, min_inside_fraction):
    return tile_inside_fraction(mask, grid, x_idx, y_idx) < min_inside_fraction


def save_pair_mask(
    pair_id,
    target_image_id,
    pyramid_page_idx,
    page_width,
    page_height,
    polygon,
    display_polygon=None,
):
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pair_id": int(pair_id),
        "target_image_id": int(target_image_id),
        "pyramid_page_idx": int(pyramid_page_idx),
        "page_width": int(page_width),
        "page_height": int(page_height),
        "polygon": [[float(x), float(y)] for x, y in polygon],
    }
    if display_polygon is not None:
        payload["display_polygon"] = [[float(x), float(y)] for x, y in display_polygon]
    path = mask_json_path(pair_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_pair_mask(pair_id):
    path = mask_json_path(pair_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mask_for_page(pair_id, page_width, page_height, pyramid_page_idx=None):
    meta = load_pair_mask(pair_id)
    if meta is None:
        return None
    polygon = scaled_polygon(
        meta["polygon"],
        meta["page_width"],
        meta["page_height"],
        page_width,
        page_height,
    )
    return rasterize_polygon(polygon, page_width, page_height)


def save_mask_preview(pair_id, mask):
    preview = (mask.astype(np.uint8) * 255)
    Image.fromarray(preview).save(mask_preview_path(pair_id))
