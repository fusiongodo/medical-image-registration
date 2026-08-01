"""
Subprocess RGB compact-page builder for regWSI inputs.

Decodes one raw RGB pyramid page, downsamples to (grid*tile_w, grid*tile_h),
writes a single-page RGB TIFF, then exits so the multi-GB decode buffers are
returned to the OS.

Usage: python build_rgb_page.py <image_id> <page_idx> <grid> <tile_w> <tile_h> <out_tiff_path>
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import tifffile

import conf


def _load_tifffile(image_path: Path, page_idx: int) -> np.ndarray:
    with tifffile.TiffFile(str(image_path)) as slide:
        return slide.pages[page_idx].asarray()


def _load_openslide_level(image_path: Path, level: int) -> np.ndarray:
    import openslide

    slide = openslide.OpenSlide(str(image_path))
    try:
        lw, lh = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, (lw, lh))
        return np.asarray(region.convert("RGB"), dtype=np.uint8)
    finally:
        slide.close()


def _candidate_pages(image_path: Path, preferred: int) -> list[int]:
    with tifffile.TiffFile(str(image_path)) as slide:
        n = len(slide.pages)
        # Prefer requested page, then coarser pyramid pages (higher index), then finer.
        rest = [i for i in range(n) if i != preferred]
        coarser = [i for i in rest if i > preferred]
        finer = [i for i in rest if i < preferred]
        return [preferred, *coarser, *finer]


def _load_page(image_path: Path, page_idx: int) -> np.ndarray:
    errors: list[str] = []
    for idx in _candidate_pages(image_path, page_idx):
        try:
            arr = _load_tifffile(image_path, idx)
            if idx != page_idx:
                print(f"using tifffile page {idx} (preferred {page_idx} failed)", flush=True)
            return arr
        except Exception as e:
            errors.append(f"tiff[{idx}]: {type(e).__name__}: {e}")
    try:
        import openslide

        slide = openslide.OpenSlide(str(image_path))
        try:
            levels = list(range(slide.level_count))
        finally:
            slide.close()
        for level in levels:
            try:
                arr = _load_openslide_level(image_path, level)
                print(f"using OpenSlide level {level} (tifffile pages failed)", flush=True)
                return arr
            except Exception as e:
                errors.append(f"openslide[{level}]: {type(e).__name__}: {e}")
    except Exception as e:
        errors.append(f"openslide: {type(e).__name__}: {e}")
    raise RuntimeError(" ; ".join(errors[-6:]))


def main() -> None:
    image_id = int(sys.argv[1])
    page_idx = int(sys.argv[2])
    grid = int(sys.argv[3])
    tile_w = int(sys.argv[4])
    tile_h = int(sys.argv[5])
    out_path = Path(sys.argv[6])

    image_path = Path(conf.resolve(conf.image_relpath(image_id)))
    raw = _load_page(image_path, page_idx)
    h, w = raw.shape[:2]
    tile_h_wsi = h // grid
    tile_w_wsi = w // grid
    raw = raw[: grid * tile_h_wsi, : grid * tile_w_wsi]
    out_w = grid * tile_w
    out_h = grid * tile_h
    compact = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    del raw
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        out_path,
        np.ascontiguousarray(compact, dtype=np.uint8),
        photometric="rgb",
        compression="zlib",
    )


if __name__ == "__main__":
    main()
