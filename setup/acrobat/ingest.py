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
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from regWSI import paths as regwsi_paths
from setup import datasets

Image.MAX_IMAGE_PIXELS = None

_NAME_RE = re.compile(
    r"^(?P<case>\d+)_(?P<stain>HE|ER|PGR|HER2|KI67)_val\.(?:tif|tiff|ndpi)$",
    re.IGNORECASE,
)
_IHC = {"ER", "PGR", "HER2", "KI67"}


class SlideReadError(RuntimeError):
    pass


def _zip_member(filename: str) -> str:
    name = Path(filename).name
    return f"valid/{name}"


def _zip_file_size(zip_path: Path, filename: str) -> int | None:
    if not zip_path.is_file():
        return None
    member = _zip_member(filename)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return int(zf.getinfo(member).file_size)
    except KeyError:
        return None


def raw_file_ok(path: Path, zip_path: Path | None = None) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    zip_path = zip_path or datasets.ACROBAT_ZIP
    expected = _zip_file_size(zip_path, path.name)
    if expected is not None and path.stat().st_size != expected:
        return False
    return True


def reextract_raw_file(filename: str, zip_path: Path | None = None, dest: Path | None = None) -> Path:
    zip_path = zip_path or datasets.ACROBAT_ZIP
    dest = dest or datasets.ACROBAT_RAW
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing {zip_path}")
    dest.mkdir(parents=True, exist_ok=True)
    member = _zip_member(filename)
    out = dest / Path(filename).name
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member) as src, out.open("wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    if not raw_file_ok(out, zip_path):
        raise RuntimeError(f"re-extract failed size check for {out.name}")
    return out


def repair_raw_from_zip(zip_path: Path | None = None, dest: Path | None = None) -> list[str]:
    zip_path = zip_path or datasets.ACROBAT_ZIP
    dest = dest or datasets.ACROBAT_RAW
    if not zip_path.is_file():
        return []
    dest.mkdir(parents=True, exist_ok=True)
    fixed: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.startswith("valid/"):
                continue
            name = Path(info.filename).name
            if not _NAME_RE.match(name):
                continue
            path = dest / name
            if raw_file_ok(path, zip_path):
                continue
            print(f"stage=reextract file={name}", flush=True)
            reextract_raw_file(name, zip_path=zip_path, dest=dest)
            fixed.append(name)
    return fixed


def ensure_unzipped(zip_path: Path | None = None, dest: Path | None = None) -> Path:
    zip_path = zip_path or datasets.ACROBAT_ZIP
    dest = dest or datasets.ACROBAT_RAW
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing {zip_path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_dir() or not any(dest.glob("*_HE_val.tif*")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest.parent)
    repair_raw_from_zip(zip_path=zip_path, dest=dest)
    if not dest.is_dir():
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


def _read_openslide(path: Path, max_side: int) -> tuple[np.ndarray, dict]:
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
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
    finally:
        slide.close()


def _read_pil(path: Path, max_side: int) -> tuple[np.ndarray, dict]:
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


def _read_rgb_level(path: Path, max_side: int) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise SlideReadError(f"missing {path}")
    if not raw_file_ok(path):
        raise SlideReadError(
            f"truncated or incomplete raw file {path.name} "
            f"(size={path.stat().st_size})"
        )
    errors: list[str] = []
    try:
        return _read_openslide(path, max_side)
    except Exception as e:
        errors.append(f"openslide: {e}")
    try:
        return _read_pil(path, max_side)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        errors.append(f"pil: {e}")
    raise SlideReadError(f"cannot read {path.name}: {'; '.join(errors)}")


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


def _cleanup_partial(out_dir: Path) -> None:
    for name in ("he.tiff", "ihc.tiff", "meta.json"):
        path = out_dir / name
        if path.is_file():
            path.unlink()


def _ensure_raw_readable(raw_dir: Path, filename: str) -> Path:
    path = raw_dir / filename
    if raw_file_ok(path):
        return path
    print(f"stage=reextract file={filename}", flush=True)
    return reextract_raw_file(filename, dest=raw_dir)


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

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            he_path = _ensure_raw_readable(raw_dir, pair["he_file"])
            ihc_path = _ensure_raw_readable(raw_dir, pair["ihc_file"])
            he_rgb, he_src = _read_rgb_level(he_path, max_side)
            ihc_rgb, ihc_src = _read_rgb_level(ihc_path, max_side)
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
        except Exception as e:
            last_err = e
            _cleanup_partial(out_dir)
            if attempt == 0:
                print(
                    f"stage=export_retry pair={pair_id} err={type(e).__name__}:{e}",
                    flush=True,
                )
                for fname in (pair["he_file"], pair["ihc_file"]):
                    try:
                        reextract_raw_file(fname, dest=raw_dir)
                    except Exception as re_err:
                        print(
                            f"stage=reextract_fail file={fname} err={re_err}",
                            flush=True,
                        )
                continue
            break

    raise SlideReadError(
        f"pair {pair_id} export failed after retry: {last_err}"
    ) from last_err


def _export_pair_job(pair: dict, force: bool) -> dict:
    pair_id = int(pair["id"])
    try:
        meta = export_pair_tiffs(pair, force=force)
        return {"ok": True, "pair_id": pair_id, "meta": meta}
    except Exception as e:
        return {
            "ok": False,
            "pair_id": pair_id,
            "case_id": pair.get("case_id"),
            "error": f"{type(e).__name__}: {e}",
        }


def ingest(
    *,
    unzip: bool = True,
    pair_ids: list[int] | None = None,
    force: bool = False,
    workers: int = 1,
) -> dict:
    if unzip:
        ensure_unzipped()
    else:
        repair_raw_from_zip()
    pairs = discover_pairs()
    if not pairs:
        raise RuntimeError(f"no HE/IHC pairs under {datasets.ACROBAT_RAW}")
    write_pairs_json(pairs)
    selected = pairs
    if pair_ids is not None:
        want = set(pair_ids)
        selected = [p for p in pairs if int(p["id"]) in want]
    metas = []
    errors: list[dict] = []
    n_workers = max(1, min(int(workers), len(selected) or 1))
    print(
        f"stage=export_start n={len(selected)} workers={n_workers} force={int(bool(force))}",
        flush=True,
    )

    if n_workers <= 1:
        for p in selected:
            pair_id = int(p["id"])
            print(f"stage=export pair={pair_id}", flush=True)
            result = _export_pair_job(p, force)
            if result.get("ok"):
                metas.append(result["meta"])
            else:
                print(
                    f"stage=export_skip pair={pair_id} err={result.get('error', '')}",
                    flush=True,
                )
                errors.append(
                    {
                        "pair_id": pair_id,
                        "case_id": result.get("case_id"),
                        "error": result.get("error"),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_export_pair_job, p, force): int(p["id"]) for p in selected
            }
            for fut in as_completed(futures):
                pair_id = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(
                        f"stage=export_skip pair={pair_id} err={type(e).__name__}:{e}",
                        flush=True,
                    )
                    errors.append(
                        {
                            "pair_id": pair_id,
                            "case_id": None,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue
                if result.get("ok"):
                    print(f"stage=export pair={pair_id}", flush=True)
                    metas.append(result["meta"])
                else:
                    print(
                        f"stage=export_skip pair={pair_id} err={result.get('error', '')}",
                        flush=True,
                    )
                    errors.append(
                        {
                            "pair_id": pair_id,
                            "case_id": result.get("case_id"),
                            "error": result.get("error"),
                        }
                    )

    return {
        "ok": len(errors) == 0,
        "n_pairs": len(pairs),
        "exported": len(metas),
        "failed": len(errors),
        "errors": errors,
        "workers": n_workers,
        "pairs_json": str(datasets.ACROBAT_PAIRS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unzip-only", action="store_true")
    ap.add_argument("--no-unzip", action="store_true")
    ap.add_argument("--pairs", default=None, help="comma-separated pair ids")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel pair exporters (unzip/repair stay single-process)",
    )
    ap.add_argument("--repair-only", action="store_true", help="re-extract truncated raw files")
    args = ap.parse_args()
    if args.repair_only:
        fixed = repair_raw_from_zip()
        print(json.dumps({"ok": True, "reextracted": fixed}))
        return
    if args.unzip_only:
        p = ensure_unzipped()
        print(json.dumps({"ok": True, "raw": str(p)}))
        return
    pair_ids = None
    if args.pairs:
        pair_ids = [int(x) for x in args.pairs.split(",") if x.strip() != ""]
    result = ingest(
        unzip=not args.no_unzip,
        pair_ids=pair_ids,
        force=args.force,
        workers=int(args.workers),
    )
    print(json.dumps(result, indent=2))
    if result.get("failed"):
        sys.exit(2)


if __name__ == "__main__":
    main()
