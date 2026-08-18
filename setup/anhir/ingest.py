"""
Ingest ANHIR medium training pairs into pipeline canvas TIFFs + landmarks.json.

Usage:
  python -m setup.anhir.ingest --pairs 0-4
  python -m setup.anhir.ingest --pairs-json-only
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from regWSI import paths as regwsi_paths
from setup import datasets
from setup.acrobat.ingest import _fit_canvas

Image.MAX_IMAGE_PIXELS = None
INDEX_CSV = "dataset_medium.csv"


def parse_pairs_spec(spec: str | None) -> list[int] | None:
    if spec is None or not str(spec).strip():
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen: set[int] = set()
    uniq: list[int] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _open_zip(zip_path: Path | None = None) -> zipfile.ZipFile:
    path = zip_path or datasets.ANHIR_ZIP
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return zipfile.ZipFile(path)


def _read_index(zf: zipfile.ZipFile) -> list[dict]:
    raw = zf.read(INDEX_CSV).decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def _case_name(image_path: str) -> str:
    return image_path.split("/", 1)[0]


def _scale_tag(image_path: str) -> str:
    parts = image_path.split("/")
    return parts[1] if len(parts) > 1 else ""


def discover_training_pairs(zf: zipfile.ZipFile | None = None) -> list[dict]:
    close = False
    if zf is None:
        zf = _open_zip()
        close = True
    try:
        names = set(zf.namelist())
        rows = _read_index(zf)
        pairs: list[dict] = []
        for row in rows:
            if (row.get("status") or "").strip().lower() != "training":
                continue
            src_im = row["Source image"]
            tgt_im = row["Target image"]
            src_lm = row["Source landmarks"]
            tgt_lm = row["Target landmarks"]
            if src_lm not in names or tgt_lm not in names:
                continue
            if src_im not in names or tgt_im not in names:
                continue
            size = row.get("Image size [pixels]") or ""
            try:
                wh = ast.literal_eval(size)
                width, height = int(wh[0]), int(wh[1])
            except Exception:
                width, height = None, None
            diag = row.get("Image diagonal [pixels]")
            try:
                diagonal = float(diag) if diag else None
            except Exception:
                diagonal = None
            pair_id = len(pairs)
            pairs.append(
                {
                    "id": pair_id,
                    "csv_index": int(row.get("", pair_id) or pair_id),
                    "case": _case_name(src_im),
                    "scale": _scale_tag(src_im),
                    "source_image": src_im,
                    "target_image": tgt_im,
                    "source_landmarks": src_lm,
                    "target_landmarks": tgt_lm,
                    "width": width,
                    "height": height,
                    "diagonal": diagonal,
                    "status": "training",
                }
            )
        return pairs
    finally:
        if close:
            zf.close()


def write_pairs_json(pairs: list[dict], path: Path | None = None) -> Path:
    path = path or datasets.ANHIR_PAIRS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": "anhir", "pairs": pairs}, indent=2))
    return path


def _parse_xy_csv(text: str) -> list[tuple[float, float]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[tuple[float, float]] = []
    for row in rows:
        keymap = {k.strip().lstrip("\ufeff"): k for k in row}
        xk = keymap.get("X") or keymap.get("x")
        yk = keymap.get("Y") or keymap.get("y")
        if xk is None or yk is None:
            raise ValueError(f"landmark csv missing X/Y columns: {list(row)}")
        out.append((float(row[xk]), float(row[yk])))
    return out


def _px_to_norm(
    x: float, y: float, fit: dict, canvas_w: int, canvas_h: int
) -> list[float]:
    downsample = max(float(fit.get("downsample") or 1.0), 1e-12)
    scale = float(fit["scale"])
    cx = float(fit["offset_x"]) + (x / downsample) * scale
    cy = float(fit["offset_y"]) + (y / downsample) * scale
    return [cx / canvas_w, cy / canvas_h]


def _read_zip_rgb(zf: zipfile.ZipFile, member: str, max_side: int) -> tuple[np.ndarray, dict]:
    with zf.open(member) as src:
        data = src.read()
    with Image.open(io.BytesIO(data)) as im:
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
        "loader": "zip-pil",
        "member": member,
    }


def export_pair(
    pair: dict,
    zf: zipfile.ZipFile,
    *,
    force: bool = False,
) -> dict:
    pair_id = int(pair["id"])
    out_dir = datasets.pair_dir(pair_id, "anhir")
    he_out = out_dir / "he.tiff"
    ihc_out = out_dir / "ihc.tiff"
    meta_path = out_dir / "meta.json"
    lm_path = out_dir / "landmarks.json"
    if (
        he_out.is_file()
        and ihc_out.is_file()
        and meta_path.is_file()
        and lm_path.is_file()
        and not force
    ):
        return json.loads(meta_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = regwsi_paths.CANVAS_W, regwsi_paths.CANVAS_H
    max_side = max(canvas_w, canvas_h)

    he_rgb, he_src = _read_zip_rgb(zf, pair["target_image"], max_side)
    ihc_rgb, ihc_src = _read_zip_rgb(zf, pair["source_image"], max_side)
    he_canvas, he_fit = _fit_canvas(he_rgb, canvas_w, canvas_h)
    ihc_canvas, ihc_fit = _fit_canvas(ihc_rgb, canvas_w, canvas_h)
    he_meta = {**he_src, **he_fit}
    ihc_meta = {**ihc_src, **ihc_fit}

    Image.fromarray(he_canvas).save(he_out, compression="tiff_lzw")
    Image.fromarray(ihc_canvas).save(ihc_out, compression="tiff_lzw")

    src_xy = _parse_xy_csv(zf.read(pair["source_landmarks"]).decode("utf-8"))
    tgt_xy = _parse_xy_csv(zf.read(pair["target_landmarks"]).decode("utf-8"))
    n = min(len(src_xy), len(tgt_xy))
    points = []
    for (sx, sy), (tx, ty) in zip(src_xy[:n], tgt_xy[:n]):
        points.append(
            {
                "he": _px_to_norm(tx, ty, he_meta, canvas_w, canvas_h),
                "ihc": _px_to_norm(sx, sy, ihc_meta, canvas_w, canvas_h),
            }
        )
    landmarks = {
        "pair_id": pair_id,
        "dataset": "anhir",
        "identity": datasets.pair_fingerprint(pair_id, "anhir"),
        "n": len(points),
        "n_source": len(src_xy),
        "n_target": len(tgt_xy),
        "points": points,
    }
    lm_path.write_text(json.dumps(landmarks, separators=(",", ":")))

    meta = {
        "dataset": "anhir",
        "pair_id": pair_id,
        "case": pair.get("case"),
        "scale": pair.get("scale"),
        "source_image": pair["source_image"],
        "target_image": pair["target_image"],
        "canvas": [canvas_w, canvas_h],
        "he": he_meta,
        "ihc": ihc_meta,
        "n_landmarks": len(points),
        "identity": datasets.pair_fingerprint(pair_id, "anhir"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def ingest(*, pair_ids: list[int] | None = None, force: bool = False) -> dict:
    with _open_zip() as zf:
        pairs = discover_training_pairs(zf)
        if not pairs:
            raise RuntimeError(f"no training pairs in {datasets.ANHIR_ZIP}")
        write_pairs_json(pairs)
        selected = pairs
        if pair_ids is not None:
            want = set(pair_ids)
            selected = [p for p in pairs if int(p["id"]) in want]
        metas = []
        errors: list[dict] = []
        print(f"stage=export_start n={len(selected)} of {len(pairs)}", flush=True)
        for p in selected:
            pair_id = int(p["id"])
            print(f"stage=export pair={pair_id} case={p.get('case')}", flush=True)
            try:
                metas.append(export_pair(p, zf, force=force))
            except Exception as e:
                print(f"stage=export_skip pair={pair_id} err={type(e).__name__}:{e}", flush=True)
                errors.append(
                    {
                        "pair_id": pair_id,
                        "case": p.get("case"),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
        return {
            "ok": len(errors) == 0,
            "n_pairs": len(pairs),
            "exported": len(metas),
            "failed": len(errors),
            "errors": errors,
            "pairs_json": str(datasets.ANHIR_PAIRS),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=None, help="comma-separated or ranges, e.g. 0-4")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--pairs-json-only",
        action="store_true",
        help="write pairs.json for all 230 training rows without extracting images",
    )
    args = ap.parse_args()
    if args.pairs_json_only:
        with _open_zip() as zf:
            pairs = discover_training_pairs(zf)
        path = write_pairs_json(pairs)
        print(json.dumps({"ok": True, "n_pairs": len(pairs), "pairs_json": str(path)}, indent=2))
        return
    result = ingest(pair_ids=parse_pairs_spec(args.pairs), force=bool(args.force))
    print(json.dumps(result, indent=2))
    if result.get("failed"):
        sys.exit(2)


if __name__ == "__main__":
    main()
