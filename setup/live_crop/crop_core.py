"""
Live tile cropping from original WSI TIFF sources.

Deterministic quadtree geometry + tissue mask only; no dependency on
{OS}_quadtree_annotations.json. Uses the polygon mask helpers (pair_mask).

Displacements (dx, dy) are given in CNN tile-pixel units (512x344 space,
matching the smooth-field / elastix convention). Each pyramid page is cached
pre-downsampled to grid*CNN greyscale, so a tile crop is a plain CNN-sized
slice and dx/dy are applied directly in CNN pixels (a positive dx yields an
IHC crop shifted right into registration, same sign convention as crop_tile).

pair_image_ids(pair_id)                 -> (he_id, ihc_id)
choose_page(pair_id, level)             -> (page_idx, tile_h, tile_w) | None
tissue_tiles(pair_id, level)            -> dict(grid, page, tile_w, tile_h, tiles=list["x_y"])
crop_png(pair_id, level, x, y, side, dx, dy)  -> bytes (PNG)
crop_gray(pair_id, level, x, y, side, dx, dy) -> np.uint8 (cnn_h, cnn_w)
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup"))
sys.path.insert(0, str(REPO_ROOT / "setup" / "labelme"))

import conf

import pair_mask

CNN_W = conf.CNN_INPUT_WIDTH
CNN_H = conf.CNN_INPUT_HEIGHT

PAD_FILL = 255

DESKEW_DIR = conf.PROJECT_ROOT / "data" / "deskew"

# Pages are cached as grid*CNN greyscale (~180 MB/side at level 5) rather than
# the raw multi-GB RGB page, so cap the warm cache by total bytes to keep the
# worker (and the run.py candidate subprocess) memory bounded.
MAX_CACHE_BYTES = 1 * 1024 ** 3

_labels_cache = None
_labels_mtime: float | None = None
_page_choice_cache: dict[tuple[int, int], tuple[int, int, int] | None] = {}
_mask_cache: dict[int, "np.ndarray | None"] = {}
_page_cache: "OrderedDict[tuple[int, int, int], np.ndarray]" = OrderedDict()
# pair_id -> (json mtime | None, affine coeffs | None); self-invalidates on mtime.
_deskew_cache: dict[int, tuple[float | None, "tuple | None"]] = {}


def _labels() -> list[dict]:
    """Pair→image-id mapping, reloaded whenever the labels file changes on disk.

    The worker is long-lived, so a stale in-memory copy would keep serving the
    old pair set after fetch_labels regenerates the file. Watching the mtime
    lets crops pick up regenerated labels without a manual worker restart.
    """
    global _labels_cache, _labels_mtime
    path = Path(conf.LABELS_PATH)
    mtime = path.stat().st_mtime
    if _labels_cache is None or mtime != _labels_mtime:
        _labels_cache = json.loads(path.read_text())
        _labels_mtime = mtime
        # These caches are keyed by pair id, whose meaning depends on the
        # labels ordering; the page cache is keyed by image id and stays valid.
        _page_choice_cache.clear()
        _mask_cache.clear()
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


def _page_cache_bytes() -> int:
    return sum(p.nbytes for p in _page_cache.values())


_BUILD_SCRIPT = Path(__file__).resolve().with_name("build_page.py")


def _compact_page(image_id: int, level: int, page_idx: int, grid: int) -> np.ndarray:
    """
    Greyscale pyramid page pre-downsampled to (grid*CNN_H, grid*CNN_W).

    Built in a short-lived subprocess (build_page.py): the raw RGB page decode
    is a multi-GB transient whose buffers macOS does not return to the OS after
    free, so building it inline would permanently inflate the long-lived worker.
    The child writes the compact array to a temp .npy and exits (handing all the
    transient memory back); we load only that (~180 MB at level 5) into RAM and
    delete the temp file, so nothing is persisted to disk. Cached LRU by
    (image_id, level, page_idx) and evicted by a total-byte budget.
    """
    key = (image_id, level, page_idx)
    compact = _page_cache.get(key)
    if compact is not None:
        _page_cache.move_to_end(key)
        return compact

    fd, tmp = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    try:
        subprocess.run(
            [sys.executable, str(_BUILD_SCRIPT), str(image_id), str(page_idx), str(grid), tmp],
            check=True,
        )
        # Copy into an owned in-memory array so no memmap keeps the file open.
        compact = np.array(np.load(tmp), dtype=np.uint8)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    _page_cache[key] = compact
    # Evict least-recently-used pages until within the byte budget, but always
    # keep the page just built (it may alone exceed the budget).
    while len(_page_cache) > 1 and _page_cache_bytes() > MAX_CACHE_BYTES:
        _page_cache.popitem(last=False)
    return compact


def _crop_padded(gpage: np.ndarray, x_idx: int, y_idx: int, dx: float, dy: float) -> np.ndarray:
    """
    Crop one CNN-sized tile from the compact greyscale page, padding (not
    clamping) out-of-page regions with PAD_FILL.

    dx/dy are CNN-pixel displacements: a positive dx shifts the moving crop into
    registration. At coarse levels or edge tiles the window can extend past the
    page; padding preserves the true displacement instead of collapsing it.
    """
    h, w = gpage.shape[:2]
    x0 = int(round(x_idx * CNN_W - dx))
    y0 = int(round(y_idx * CNN_H - dy))
    x1 = x0 + CNN_W
    y1 = y0 + CNN_H

    out = np.full((CNN_H, CNN_W), PAD_FILL, dtype=gpage.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gpage[sy0:sy1, sx0:sx1]
    return out


def _deskew_affine(pair_id: int) -> "tuple | None":
    """
    Global deskew affine for a pair as ((a0,a1,s),(b0,s,b2)) in normalised [0,1]
    image space, or None. Read from data/deskew/{pair}.json and cached with an
    mtime guard so edits (or clears) are picked up without a worker restart.
    """
    path = DESKEW_DIR / f"{pair_id}.json"
    mtime = path.stat().st_mtime if path.exists() else None
    cached = _deskew_cache.get(pair_id)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    affine = None
    if mtime is not None:
        try:
            affine = json.loads(path.read_text()).get("affine")
        except Exception:
            affine = None
    _deskew_cache[pair_id] = (mtime, affine)
    return affine


def _crop_warped(gpage: np.ndarray, x_idx: int, y_idx: int, dx: float, dy: float, affine) -> np.ndarray:
    """
    Crop one CNN-sized moving tile while resampling the whole page through the
    global deskew affine, so a strong stretch/shear is corrected within the tile
    (a per-tile translation cannot do this). dx/dy add the residual FFT/field
    offset on top, exactly as _crop_padded does.

    The affine d(p)=c+L·p is the normalised HE-minus-IHC displacement, so the
    aligned moving sample for output position p is ihc(p - d(p)) = ihc((I-L)p-c).
    In this page's pixel space (W=grid*CNN_W, H=grid*CNN_H) that is the inverse
    map fed to ndimage.affine_transform (row=y, col=x order).
    """
    h, w = gpage.shape[:2]
    (a0, a1, a2), (b0, _b1, b2) = affine
    s = a2
    wx0 = x_idx * CNN_W - dx
    wy0 = y_idx * CNN_H - dy
    m00 = 1.0 - b2
    m01 = -s * (h / w)
    m10 = -s * (w / h)
    m11 = 1.0 - a1
    matrix = np.array([[m00, m01], [m10, m11]], dtype=float)
    off_row = m00 * wy0 + m01 * wx0 - b0 * h
    off_col = m10 * wy0 + m11 * wx0 - a0 * w
    out = ndimage.affine_transform(
        gpage,
        matrix,
        offset=(off_row, off_col),
        output_shape=(CNN_H, CNN_W),
        order=1,
        mode="constant",
        cval=PAD_FILL,
    )
    return out.astype(gpage.dtype)


def _crop_tile(pair_id: int, level: int, x: int, y: int, side: str, dx: float, dy: float) -> np.ndarray:
    """np.uint8 (CNN_H, CNN_W) greyscale tile with a CNN-pixel offset.

    The moving (IHC) side is resampled through the pair's global deskew affine
    when one is stored; the fixed (HE) side is always a plain padded crop.
    """
    he_id, ihc_id = pair_image_ids(pair_id)
    image_id = he_id if side == "he" else ihc_id

    chosen = choose_page(pair_id, level)
    if chosen is None:
        raise ValueError(f"no pyramid page for pair {pair_id} level {level}")
    page_idx = chosen[0]

    grid = 2 ** level
    gpage = _compact_page(image_id, level, page_idx, grid)
    if side != "he":
        affine = _deskew_affine(pair_id)
        if affine is not None:
            return _crop_warped(gpage, x, y, dx, dy, affine)
    return _crop_padded(gpage, x, y, dx, dy)


def crop_png(
    pair_id: int, level: int, x: int, y: int, side: str, dx: float = 0.0, dy: float = 0.0
) -> bytes:
    tile = _crop_tile(pair_id, level, x, y, side, dx, dy)
    buffer = io.BytesIO()
    Image.fromarray(tile, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def crop_gray(
    pair_id: int, level: int, x: int, y: int, side: str, dx: float = 0.0, dy: float = 0.0
) -> np.ndarray:
    return _crop_tile(pair_id, level, x, y, side, dx, dy)


def whole_gray(pair_id: int, side: str, level: int) -> "np.ndarray | None":
    """
    Whole-image greyscale preview at grid*CNN resolution (raw, no deskew warp),
    used for placing deskew correspondence landmarks. `level` selects the
    resolution: level 0 is 512x344, level 2 is 2048x1376 (4x). Returns None when
    no suitable pyramid page exists.
    """
    he_id, ihc_id = pair_image_ids(pair_id)
    image_id = he_id if side == "he" else ihc_id
    chosen = choose_page(pair_id, level)
    if chosen is None:
        return None
    return _compact_page(image_id, level, chosen[0], 2 ** level)


def whole_png(pair_id: int, side: str, level: int) -> "bytes | None":
    gpage = whole_gray(pair_id, side, level)
    if gpage is None:
        return None
    buffer = io.BytesIO()
    Image.fromarray(gpage, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()
