"""
Live tile cropping from original WSI TIFF sources.

Deterministic quadtree geometry + tissue mask only; no dependency on
{OS}_quadtree_annotations.json. Reuses crop_tile / tile_to_gray_png_array
(preprocess_tiles) and the polygon mask helpers (pair_mask).

Displacements (dx, dy) are given in CNN tile-pixel units (512x344 space,
matching the smooth-field / elastix convention) and converted to WSI-page
pixels internally, so a positive dx yields an IHC crop shifted right into
registration (same sign convention as crop_tile / preprocess_tiles --smooth).

pair_image_ids(pair_id)                 -> (he_id, ihc_id)
choose_page(pair_id, level)             -> (page_idx, tile_h, tile_w) | None
tissue_tiles(pair_id, level)            -> dict(grid, page, tile_w, tile_h, tiles=list["x_y"])
crop_png(pair_id, level, x, y, side, dx, dy)  -> bytes (PNG)
crop_gray(pair_id, level, x, y, side, dx, dy) -> np.uint8 (cnn_h, cnn_w)
"""

import io
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup"))
sys.path.insert(0, str(REPO_ROOT / "setup" / "labelme"))

import conf
import pair_mask
from preprocess_tiles import crop_tile, tile_to_gray_png_array

MAX_CACHED_PAGES = 6

_labels_cache = None
_page_choice_cache: dict[tuple[int, int], tuple[int, int, int] | None] = {}
_mask_cache: dict[int, "np.ndarray | None"] = {}
_page_cache: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()


def _labels() -> list[dict]:
    global _labels_cache
    if _labels_cache is None:
        _labels_cache = json.loads(Path(conf.LABELS_PATH).read_text())
    return _labels_cache


def num_pairs() -> int:
    return len(_labels())


def pair_image_ids(pair_id: int) -> tuple[int, int]:
    item = _labels()[pair_id]
    return item["target_image_id"], item["source_image_id"]


def _image_path(image_id: int) -> Path:
    return conf.resolve(conf.image_relpath(image_id))


def choose_page(pair_id: int, level: int) -> tuple[int, int, int] | None:
    key = (pair_id, level)
    if key in _page_choice_cache:
        return _page_choice_cache[key]

    he_id, ihc_id = pair_image_ids(pair_id)
    grid = 2 ** level
    chosen = None
    with (
        tifffile.TiffFile(_image_path(he_id)) as fixed_slide,
        tifffile.TiffFile(_image_path(ihc_id)) as moving_slide,
    ):
        for page_idx in conf.WSI_PAGES:
            fh, fw = fixed_slide.pages[page_idx].shape[:2]
            mh, mw = moving_slide.pages[page_idx].shape[:2]
            min_tile_h = min(fh // grid, mh // grid)
            min_tile_w = min(fw // grid, mw // grid)
            if min_tile_h >= conf.CNN_INPUT_HEIGHT and min_tile_w >= conf.CNN_INPUT_WIDTH:
                chosen = (page_idx, min_tile_h, min_tile_w)

    _page_choice_cache[key] = chosen
    return chosen


def _pair_mask(pair_id: int) -> "np.ndarray | None":
    """
    Boolean tissue mask (H, W) rasterised once at the mask's native (small)
    page resolution. Tile inside-fractions are computed by slicing this array,
    so exclusion cost is O(grid^2) and independent of the crop pyramid level.
    """
    if pair_id in _mask_cache:
        return _mask_cache[pair_id]

    meta = pair_mask.load_pair_mask(pair_id)
    if meta is None:
        _mask_cache[pair_id] = None
        return None

    mask = pair_mask.rasterize_polygons(
        pair_mask.mask_polygons(meta),
        meta["page_width"],
        meta["page_height"],
    )
    _mask_cache[pair_id] = mask
    return mask


def tissue_tiles(pair_id: int, level: int) -> dict:
    grid = 2 ** level
    chosen = choose_page(pair_id, level)
    if chosen is None:
        return {"grid": grid, "page": None, "tile_w": 0, "tile_h": 0, "tiles": []}

    page_idx, tile_h, tile_w = chosen
    mask = _pair_mask(pair_id)
    threshold = pair_mask.PRODUCTION_MIN_INSIDE_FRACTION

    tiles: list[str] = []
    for y in range(grid):
        for x in range(grid):
            if mask is None:
                excluded = False
            else:
                excluded = pair_mask.tile_inside_fraction(mask, grid, x, y) < threshold
            if not excluded:
                tiles.append(f"{x}_{y}")

    return {
        "grid": grid,
        "page": page_idx,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "tiles": tiles,
    }


def _load_page(image_id: int, page_idx: int) -> np.ndarray:
    key = (image_id, page_idx)
    page = _page_cache.get(key)
    if page is None:
        with tifffile.TiffFile(_image_path(image_id)) as slide:
            page = slide.pages[page_idx].asarray()
        _page_cache[key] = page
        while len(_page_cache) > MAX_CACHED_PAGES:
            _page_cache.popitem(last=False)
    else:
        _page_cache.move_to_end(key)
    return page


def _crop_pil(pair_id: int, level: int, x: int, y: int, side: str, dx: float, dy: float):
    """Return a PIL Image (mode 'L', cnn_w x cnn_h) for one tile with a tile-pixel offset."""
    he_id, ihc_id = pair_image_ids(pair_id)
    image_id = he_id if side == "he" else ihc_id

    chosen = choose_page(pair_id, level)
    if chosen is None:
        raise ValueError(f"no pyramid page for pair {pair_id} level {level}")
    page_idx = chosen[0]

    page = _load_page(image_id, page_idx)
    grid = 2 ** level
    tile_w_wsi = page.shape[1] // grid
    tile_h_wsi = page.shape[0] // grid
    dx_wsi = dx * tile_w_wsi / conf.CNN_INPUT_WIDTH
    dy_wsi = dy * tile_h_wsi / conf.CNN_INPUT_HEIGHT

    tile = crop_tile(page, x, y, grid, dx_wsi, dy_wsi)
    return tile_to_gray_png_array(tile)


def crop_png(
    pair_id: int, level: int, x: int, y: int, side: str, dx: float = 0.0, dy: float = 0.0
) -> bytes:
    buffer = io.BytesIO()
    _crop_pil(pair_id, level, x, y, side, dx, dy).save(buffer, format="PNG")
    return buffer.getvalue()


def crop_gray(
    pair_id: int, level: int, x: int, y: int, side: str, dx: float = 0.0, dy: float = 0.0
) -> np.ndarray:
    return np.asarray(_crop_pil(pair_id, level, x, y, side, dx, dy), dtype=np.uint8)
