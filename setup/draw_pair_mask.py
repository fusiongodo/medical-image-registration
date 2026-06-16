import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.widgets import PolygonSelector
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf

from setup.pair_mask import (
    mask_json_path,
    save_mask_preview,
    save_pair_mask,
)

PAIR_ID = 0
PYRAMID_PAGE_IDX = 4
DISPLAY_MAX_SIDE = 8192
FIGSIZE = (24, 18)


def load_pair_target(pair_id):
    with open(conf.LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    item = labels[pair_id]
    target_id = item["target_image_id"]
    path = conf.resolve(conf.image_relpath(target_id))
    if not path.exists():
        raise FileNotFoundError(path)
    return target_id, path


def load_page_gray(path, pyramid_page_idx):
    with tifffile.TiffFile(path) as slide:
        page = slide.pages[pyramid_page_idx].asarray()
    if page.ndim == 3:
        page = np.dot(page[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif page.dtype != np.uint8:
        page = page.astype(np.uint8)
    return page


def downsample_for_display(page, max_side):
    height, width = page.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return page, 1.0
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = Image.fromarray(page).convert("L")
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(image, dtype=np.uint8), scale


def display_to_page_coords(verts, scale):
    inv = 1.0 / scale
    return [[x * inv, y * inv] for x, y in verts]


def main():
    target_id, path = load_pair_target(PAIR_ID)
    page = load_page_gray(path, PYRAMID_PAGE_IDX)
    page_h, page_w = page.shape[:2]
    display, scale = downsample_for_display(page, DISPLAY_MAX_SIDE)
    disp_h, disp_w = display.shape[:2]

    saved = {"polygon": None}

    def on_select(verts):
        if len(verts) < 3:
            return
        saved["polygon"] = display_to_page_coords(verts, scale)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.canvas.manager.set_window_title(
        f"Pair {PAIR_ID} HE mask — toolbar: zoom/pan, then draw polygon"
    )
    ax.imshow(display, cmap="gray", origin="upper", interpolation="nearest")
    ax.set_title(
        f"pair {PAIR_ID} | HE {target_id} | page {PYRAMID_PAGE_IDX} | "
        f"display {disp_w}x{disp_h} (page {page_w}x{page_h})"
    )
    ax.set_xlim(-0.5, disp_w - 0.5)
    ax.set_ylim(disp_h - 0.5, -0.5)

    PolygonSelector(
        ax,
        on_select,
        useblit=True,
        props={"color": "lime", "linewidth": 2, "alpha": 0.8},
        handle_props={"markeredgecolor": "yellow", "markerfacecolor": "lime"},
    )

    print(f"Pair ID         : {PAIR_ID}")
    print(f"HE target       : {target_id}")
    print(f"Pyramid page    : {PYRAMID_PAGE_IDX}")
    print(f"Page size       : {page_w} x {page_h}")
    print(f"Display size    : {disp_w} x {disp_h} (scale {scale:.4f})")
    print(f"Output          : {mask_json_path(PAIR_ID)}")
    print()
    print("Use the matplotlib toolbar (bottom) to zoom and pan.")
    print("Click vertices to draw the valid tissue region.")
    print("Close the polygon (click near the first point), then close the window.")
    print()

    plt.tight_layout()
    plt.show()

    polygon = saved["polygon"]
    if not polygon or len(polygon) < 3:
        print("[ERROR] No polygon saved — draw a closed region before closing the window.")
        sys.exit(1)

    out_path = save_pair_mask(
        pair_id=PAIR_ID,
        target_image_id=target_id,
        pyramid_page_idx=PYRAMID_PAGE_IDX,
        page_width=page_w,
        page_height=page_h,
        polygon=polygon,
    )

    from setup.pair_mask import rasterize_polygon

    mask = rasterize_polygon(polygon, page_w, page_h)
    save_mask_preview(PAIR_ID, mask)

    from setup.pair_mask import mask_preview_path

    print(f"Saved polygon : {out_path}")
    print(f"Preview mask  : {mask_preview_path(PAIR_ID)}")


if __name__ == "__main__":
    main()
