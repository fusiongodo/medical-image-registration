"""
Subprocess compact-page builder.

Decodes one raw RGB pyramid page (a multi-GB transient), downsamples it to
grid*CNN greyscale and writes the small result to a .npy path, then exits.

Running this as a separate process is deliberate: on macOS the decode/OpenCV
buffers are not returned to the OS after free, so doing it inline would leave
the long-lived crop worker permanently inflated. Letting a child process die
hands all that transient memory back; the parent only ever loads the compact
array (~180 MB at level 5) into RAM.

Usage: python build_page.py <image_id> <page_idx> <grid> <out_npy_path>
"""

import sys
from pathlib import Path

# Import only conf (not crop_core) to avoid pulling in pair_mask/labelme, which
# would slow this short-lived process's startup.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import tifffile

import conf


def main() -> None:
    image_id = int(sys.argv[1])
    page_idx = int(sys.argv[2])
    grid = int(sys.argv[3])
    out_path = sys.argv[4]

    image_path = conf.resolve(conf.image_relpath(image_id))
    with tifffile.TiffFile(image_path) as slide:
        raw = slide.pages[page_idx].asarray()
    h, w = raw.shape[:2]
    tile_h_wsi = h // grid
    tile_w_wsi = w // grid
    raw = raw[: grid * tile_h_wsi, : grid * tile_w_wsi]
    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    del raw
    compact = cv2.resize(
        gray,
        (grid * conf.CNN_INPUT_WIDTH, grid * conf.CNN_INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    del gray
    np.save(out_path, np.ascontiguousarray(compact, dtype=np.uint8))


if __name__ == "__main__":
    main()
