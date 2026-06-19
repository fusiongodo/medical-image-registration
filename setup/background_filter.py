from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image

import conf

POOL_GRID = 16
ANALYSIS_SIZE = 128
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98

WHITE_CUTOFF = 0.87
MONOTONE_STD_MAX = 0.06


@dataclass(frozen=True)
class FilterConfig:
    min_tissue_cell_fraction: float
    structure_std_min: float
    structure_grad_min: float
    bright_strong_mean: float
    tile_std_min: float | None = None


TEST_RUN_CONFIGS = {
    "cfg_0": FilterConfig(0.40, 0.038, 0.020, 0.75, None),
    "cfg_1": FilterConfig(0.35, 0.035, 0.018, 0.78, None),
    "cfg_2": FilterConfig(0.33, 0.032, 0.017, 0.80, None),
    "cfg_3": FilterConfig(0.28, 0.028, 0.014, 0.82, 0.14),
    "cfg_4": FilterConfig(0.20, 0.025, 0.012, 0.85, 0.08),
}

PRODUCTION_CONFIG = TEST_RUN_CONFIGS["cfg_2"]


def normalize_contrast(gray):
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    lo, hi = np.percentile(g, [PERCENTILE_LOW, PERCENTILE_HIGH])
    span = hi - lo
    if span < 1e-6:
        return np.zeros_like(g)
    return np.clip((g - lo) / span, 0.0, 1.0)


def _cell_has_tissue(patch, grad_patch, config):
    mean = float(patch.mean())
    std = float(patch.std())
    grad_mean = float(grad_patch.mean())

    if mean > WHITE_CUTOFF and std < MONOTONE_STD_MAX:
        return False

    strong = std >= config.structure_std_min and grad_mean >= config.structure_grad_min
    moderate = (
        std >= config.structure_std_min * 1.4
        or grad_mean >= config.structure_grad_min * 1.4
    )

    if mean > config.bright_strong_mean:
        return strong
    return moderate


def tissue_pass_fraction(crop, config):
    if crop.dtype != np.uint8:
        crop = crop.astype(np.uint8)
    gray = np.array(Image.fromarray(crop).convert("L"), dtype=np.float32) / 255.0
    norm = normalize_contrast(gray)
    norm_u8 = (norm * 255).astype(np.uint8)

    small = np.array(
        Image.fromarray(norm_u8).resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    gy, gx = np.gradient(small)
    grad = np.hypot(gx, gy)

    cell_h = ANALYSIS_SIZE // POOL_GRID
    cell_w = ANALYSIS_SIZE // POOL_GRID
    passing = 0

    for row in range(POOL_GRID):
        for col in range(POOL_GRID):
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = ANALYSIS_SIZE if row == POOL_GRID - 1 else y0 + cell_h
            x1 = ANALYSIS_SIZE if col == POOL_GRID - 1 else x0 + cell_w
            patch = small[y0:y1, x0:x1]
            grad_patch = grad[y0:y1, x0:x1]
            if _cell_has_tissue(patch, grad_patch, config):
                passing += 1

    return passing / (POOL_GRID * POOL_GRID)


def is_background_crop(crop, config):
    if config.tile_std_min is not None:
        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)
        gray = np.array(Image.fromarray(crop).convert("L"), dtype=np.float32) / 255.0
        norm = normalize_contrast(gray)
        if float(norm.std()) >= config.tile_std_min:
            return False
    return tissue_pass_fraction(crop, config) < config.min_tissue_cell_fraction


def is_background_tile(path, pyramid_page_idx, grid, x_idx, y_idx, page_cache, config=PRODUCTION_CONFIG):
    path = conf.resolve(path)
    key = (str(path), pyramid_page_idx)
    if key not in page_cache:
        with tifffile.TiffFile(path) as slide:
            page_cache[key] = slide.pages[pyramid_page_idx].asarray()
    page = page_cache[key]
    h, w = page.shape[:2]
    tile_w = w // grid
    tile_h = h // grid
    x0, y0 = x_idx * tile_w, y_idx * tile_h
    x1 = w if x_idx == grid - 1 else x0 + tile_w
    y1 = h if y_idx == grid - 1 else y0 + tile_h
    return is_background_crop(page[y0:y1, x0:x1], config)


def load_fixed_crop(job, page_cache):
    path = conf.job_image_path(job, "fixed")
    key = (str(conf.resolve(path)), int(job["pyramid_page_idx"]))
    if key not in page_cache:
        with tifffile.TiffFile(conf.resolve(path)) as slide:
            page_cache[key] = slide.pages[job["pyramid_page_idx"]].asarray()
    page = page_cache[key]
    grid = job["grid"]
    x_idx = job["x_idx"]
    y_idx = job["y_idx"]
    h, w = page.shape[:2]
    tile_w = w // grid
    tile_h = h // grid
    x0, y0 = x_idx * tile_w, y_idx * tile_h
    x1 = w if x_idx == grid - 1 else x0 + tile_w
    y1 = h if y_idx == grid - 1 else y0 + tile_h
    return page[y0:y1, x0:x1]
