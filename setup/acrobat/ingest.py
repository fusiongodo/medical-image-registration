"""
Ingest ACROBAT validation WSIs into pipeline working tiffs.

Usage:
  python -m setup.acrobat.ingest
  python -m setup.acrobat.ingest --unzip-only
  python -m setup.acrobat.ingest --pairs 0,1,2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import conf
from regWSI import paths as regwsi_paths
from setup import datasets

Image.MAX_IMAGE_PIXELS = None

_NAME_RE = re.compile(
    r"^(?P<case>\d+)_(?P<stain>HE|ER|PGR|HER2|KI67)_val\.(?:tif|tiff|ndpi)$",
    re.IGNORECASE,
)
_IHC = {"ER", "PGR", "HER2", "KI67"}


def ensure_unzipped(zip_path: Path | None = None, dest: Path | None = None) -> Path:
    zip_path = zip_path or datasets.ACROBAT_ZIP
    dest = dest or datasets.ACROBAT_RAW
    if dest.is_dir() and any(dest.glob("*_HE_val.tif*")):
        return dest
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing {zip_path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest.parent.parent if dest.name == "valid" else dest.parent)
    if not dest.is_dir():
        alt = datasets.ACROBAT_ROOT / "raw" / "valid"
        if alt.is_dir():
            return alt
        raise FileNotFoundError(f"unzip finished but {dest} missing")
    return dest


def discover_pairs(raw_dir: Path | None = None) -> list[dict]:
    raw_dir = raw_dir or datasets.ACROBAT_RAW
    he: dict[int, Path] = {}
    ihc: dict[int, tuple[str, Path]] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        m = _NAME_RE.match(path.name)
        if not m:
            continue
        case_id = int(m.group("case"))
        stain = m.group("stain").upper()
        if stain == "HE":
            he[case_id] = path
        elif stain in _IHC:
            ihc[case_id] = (stain, path)
    pairs = []
    for i, case_id in enumerate(sorted(set(he) & set(ihc))):
        stain, ihc_path = ihc[case_id]
        pairs.append(
            {
                "id": i,
                "case_id": case_id,
                "he_file": he[case_id].name,
                "ihc_file": ihc_path.name,
                "ihc_stain": stain,
            }
        )
    return pairs


def write_pairs_json(pairs: list[dict], path: Path | None = None) -> Path:
    path = path or datasets.ACROBAT_PAIRS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": "acrobat", "pairs": pairs}, indent=2))
    return path


def _read_rgb_level(path: Path, max_side: int) -> tuple[np.ndarray, dict]:
    try:
        import openslide

        slide = openslide.OpenSlide(str(path))
        level0_w, level0_h = slide.level_dimensions[0]
        level = slide.level_count - 1
        for li in range(slide.level_count):
            w, h = slide.level_dimensions[li]
            if max(w, h) <= max_side * 1.25:
                level = li
                break
        w, h = slide.level_dimensions[level]
        downsample = float(slide.level_downsamples[level])
        rgba = slide.read_region((0, 0), level, (w, h))
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
        mpp = None
        for key in ("openslide.mpp-x", "tiff.XResolution"):
            if key in slide.properties:
                try:
                    mpp = float(slide.properties[key])
                except Exception:
                    pass
                break
        slide.close()
        return rgb, {
            "level": int(level),
            "src_w": int(w),
            "src_h": int(h),
            "level0_w": int(level0_w),
            "level0_h": int(level0_h),
            "downsample": downsample,
            "mpp": mpp,
            "loader": "openslide",
        }
    except Exception:
        pass

    with Image.open(path) as im:
        im.seek(0)
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h0, w0 = rgb.shape[:2]
    h, w = h0, w0
    downsample = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        rgb = np.asarray(
            Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        h, w = rgb.shape[:2]
        downsample = max(w0 / float(w), h0 / float(h))
    return rgb, {
        "level": 0,
        "src_w": int(w),
        "src_h": int(h),
        "level0_w": int(w0),
        "level0_h": int(h0),
        "downsample": float(downsample),
        "mpp": None,
        "loader": "pil",
    }


def _fit_canvas(rgb: np.ndarray, canvas_w: int, canvas_h: int) -> tuple[np.ndarray, dict]:
    h, w = rgb.shape[:2]
    scale = min(canvas_w / float(w), canvas_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    out = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    x0 = (canvas_w - nw) // 2
    y0 = (canvas_h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out, {
        "scale": float(scale),
        "offset_x": int(x0),
        "offset_y": int(y0),
        "content_w": int(nw),
        "content_h": int(nh),
        "src_w": int(w),
        "src_h": int(h),
    }


def export_pair_tiffs(
    pair: dict,
    raw_dir: Path | None = None,
    force: bool = False,
) -> dict:
    raw_dir = raw_dir or datasets.ACROBAT_RAW
    pair_id = int(pair["id"])
    out_dir = datasets.pair_dir(pair_id, "acrobat")
    he_out = out_dir / "he.tiff"
    ihc_out = out_dir / "ihc.tiff"
    meta_path = out_dir / "meta.json"
    if he_out.is_file() and ihc_out.is_file() and meta_path.is_file() and not force:
        return json.loads(meta_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = regwsi_paths.CANVAS_W, regwsi_paths.CANVAS_H
    max_side = max(canvas_w, canvas_h)

    he_rgb, he_src = _read_rgb_level(raw_dir / pair["he_file"], max_side)
    ihc_rgb, ihc_src = _read_rgb_level(raw_dir / pair["ihc_file"], max_side)
    he_canvas, he_fit = _fit_canvas(he_rgb, canvas_w, canvas_h)
    ihc_canvas, ihc_fit = _fit_canvas(ihc_rgb, canvas_w, canvas_h)

    Image.fromarray(he_canvas).save(he_out, compression="tiff_lzw")
    Image.fromarray(ihc_canvas).save(ihc_out, compression="tiff_lzw")

    meta = {
        "dataset": "acrobat",
        "pair_id": pair_id,
        "case_id": pair["case_id"],
        "ihc_stain": pair["ihc_stain"],
        "he_file": pair["he_file"],
        "ihc_file": pair["ihc_file"],
        "canvas": [canvas_w, canvas_h],
        "he": {**he_src, **he_fit},
        "ihc": {**ihc_src, **ihc_fit},
        "identity": datasets.pair_fingerprint(pair_id, "acrobat"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def ingest(
    *,
    unzip: bool = True,
    pair_ids: list[int] | None = None,
    force: bool = False,
) -> dict:
    if unzip:
        ensure_unzipped()
    pairs = discover_pairs()
    if not pairs:
        raise RuntimeError(f"no HE/IHC pairs under {datasets.ACROBAT_RAW}")
    write_pairs_json(pairs)
    selected = pairs
    if pair_ids is not None:
        want = set(pair_ids)
        selected = [p for p in pairs if int(p["id"]) in want]
    metas = []
    for p in selected:
        metas.append(export_pair_tiffs(p, force=force))
    return {
        "ok": True,
        "n_pairs": len(pairs),
        "exported": len(metas),
        "pairs_json": str(datasets.ACROBAT_PAIRS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unzip-only", action="store_true")
    ap.add_argument("--no-unzip", action="store_true")
    ap.add_argument("--pairs", default=None, help="comma-separated pair ids")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.unzip_only:
        p = ensure_unzipped()
        print(json.dumps({"ok": True, "raw": str(p)}))
        return
    pair_ids = None
    if args.pairs:
        pair_ids = [int(x) for x in args.pairs.split(",") if x.strip() != ""]
    print(
        json.dumps(
            ingest(unzip=not args.no_unzip, pair_ids=pair_ids, force=args.force),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
