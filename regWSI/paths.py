"""
Path helpers for DeeperHistReg (regWSI) preregistration artifacts.

Layout under data/regwsi/{pair}/:
  he.tiff / ihc.tiff          RGB level-5 canvas inputs (SCALE x CNN tile)
  out/displacement_field.mha  composed affine+deformable field ("the sum")
  out/warped_ihc.tiff         IHC warped into HE space
  out/target.tiff             HE copy (from deeperhistreg)
  preview/he.png              browser-friendly HE
  preview/ihc_warped.png      browser-friendly warped IHC
  full/{layer}_y{qy}_x{qx}.jpg  FULL_NQ x FULL_NQ explorer mosaic
  full/meta.json              {w,h,qw,qh,nq}
  landmarks.json              HE/IHC correspondence points in [0,1]
  meta.json                   identity + canvas + params stamp
"""

from __future__ import annotations

from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import conf

LEVEL = 5
GRID = 2 ** LEVEL
SCALE = 1
CANVAS_W = SCALE * GRID * conf.CNN_INPUT_WIDTH
CANVAS_H = SCALE * GRID * conf.CNN_INPUT_HEIGHT
TILE_W = SCALE * conf.CNN_INPUT_WIDTH
TILE_H = SCALE * conf.CNN_INPUT_HEIGHT

# Corrupt JPEG tiles on the L5 pyramid page — skip for fair 1x regWSI batch.
EXCLUDED_PAIR_IDS = frozenset({15, 19, 20})

REGWSI_ROOT = conf.PROJECT_ROOT / "data" / "regwsi"
PREVIEW_MAX_SIDE = 2048
FULL_NQ = 2
FULL_LAYERS = ("he", "ihc", "ihc_warped")
FULL_QUAD_COORDS = tuple((qy, qx) for qy in range(FULL_NQ) for qx in range(FULL_NQ))


def pair_dir(pair_id: int) -> Path:
    return REGWSI_ROOT / str(pair_id)


def he_tiff(pair_id: int) -> Path:
    return pair_dir(pair_id) / "he.tiff"


def ihc_tiff(pair_id: int) -> Path:
    return pair_dir(pair_id) / "ihc.tiff"


def out_dir(pair_id: int) -> Path:
    return pair_dir(pair_id) / "out"


def displacement_field(pair_id: int) -> Path:
    return out_dir(pair_id) / "displacement_field.mha"


def warped_ihc(pair_id: int) -> Path:
    return out_dir(pair_id) / "warped_ihc.tiff"


def preview_dir(pair_id: int) -> Path:
    return pair_dir(pair_id) / "preview"


def preview_he(pair_id: int) -> Path:
    return preview_dir(pair_id) / "he.png"


def preview_ihc_warped(pair_id: int) -> Path:
    return preview_dir(pair_id) / "ihc_warped.png"


def meta_json(pair_id: int) -> Path:
    return pair_dir(pair_id) / "meta.json"


def landmarks_json(pair_id: int) -> Path:
    return pair_dir(pair_id) / "landmarks.json"


def full_dir(pair_id: int) -> Path:
    return pair_dir(pair_id) / "full"


def full_meta_json(pair_id: int) -> Path:
    return full_dir(pair_id) / "meta.json"


def full_quadrant(pair_id: int, layer: str, qy: int, qx: int) -> Path:
    return full_dir(pair_id) / f"{layer}_y{qy}_x{qx}.jpg"


def ensure_pair_dirs(pair_id: int) -> None:
    pair_dir(pair_id).mkdir(parents=True, exist_ok=True)
    out_dir(pair_id).mkdir(parents=True, exist_ok=True)
    preview_dir(pair_id).mkdir(parents=True, exist_ok=True)
    full_dir(pair_id).mkdir(parents=True, exist_ok=True)


def is_full_ready(pair_id: int) -> bool:
    if not full_meta_json(pair_id).is_file():
        return False
    try:
        import json

        meta = json.loads(full_meta_json(pair_id).read_text())
        nq = int(meta.get("nq", FULL_NQ))
    except Exception:
        nq = FULL_NQ
    coords = [(qy, qx) for qy in range(nq) for qx in range(nq)]
    return all(
        full_quadrant(pair_id, layer, qy, qx).is_file()
        for layer in FULL_LAYERS
        for qy, qx in coords
    )


def is_registered(pair_id: int) -> bool:
    return displacement_field(pair_id).is_file() and preview_he(pair_id).is_file() and preview_ihc_warped(pair_id).is_file()
