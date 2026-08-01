"""
Export SCALE=1 zlib RGB TIFFs (same build_rgb_page path as FFT L5 source) and
pack them into a tar.zst for VPS upload.

Usage:
  python regWSI/pack_inputs.py
  python regWSI/pack_inputs.py --pairs 0 1 2
  python regWSI/pack_inputs.py --skip-export   # pack existing 1x tiffs only
  python regWSI/pack_inputs.py -o /tmp/regwsi_inputs_1x.tar.zst
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))

import crop_core
from setup.coarse_to_fine.identity import pair_fingerprint

from regWSI import paths
from regWSI.export_slides import export_pair


def _tiff_shape(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        h, w = int(page.shape[0]), int(page.shape[1])
    return w, h


def _is_scale1(path: Path) -> bool:
    shape = _tiff_shape(path)
    if shape is None:
        return False
    return shape == (paths.CANVAS_W, paths.CANVAS_H)


def export_all(pair_ids: list[int], force: bool) -> tuple[list[dict], list[int]]:
    rows = []
    failed: list[int] = []
    for pair_id in pair_ids:
        need = force or not (_is_scale1(paths.he_tiff(pair_id)) and _is_scale1(paths.ihc_tiff(pair_id)))
        print(f"export pair {pair_id} force={need} …", flush=True)
        try:
            row = export_pair(pair_id, force=need)
            if not _is_scale1(paths.he_tiff(pair_id)) or not _is_scale1(paths.ihc_tiff(pair_id)):
                raise RuntimeError(
                    f"pair {pair_id} tiffs are not SCALE=1 "
                    f"({paths.CANVAS_W}x{paths.CANVAS_H}); got "
                    f"he={_tiff_shape(paths.he_tiff(pair_id))} ihc={_tiff_shape(paths.ihc_tiff(pair_id))}"
                )
            rows.append(row)
        except Exception as e:
            print(f"FAIL pair {pair_id}: {e}", flush=True)
            failed.append(pair_id)
    return rows, failed


def write_manifest(pair_ids: list[int], out_path: Path) -> dict:
    pairs = []
    for pair_id in pair_ids:
        he = paths.he_tiff(pair_id)
        ihc = paths.ihc_tiff(pair_id)
        pairs.append(
            {
                "pair_id": pair_id,
                "identity": pair_fingerprint(pair_id),
                "he": str(he.relative_to(REPO_ROOT)),
                "ihc": str(ihc.relative_to(REPO_ROOT)),
                "he_bytes": he.stat().st_size,
                "ihc_bytes": ihc.stat().st_size,
                "canvas": [paths.CANVAS_W, paths.CANVAS_H],
            }
        )
    manifest = {
        "scale": paths.SCALE,
        "level": paths.LEVEL,
        "canvas": [paths.CANVAS_W, paths.CANVAS_H],
        "created": datetime.now(timezone.utc).isoformat(),
        "excluded_pairs": sorted(paths.EXCLUDED_PAIR_IDS),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def pack_tar_zst(pair_ids: list[int], archive: Path, manifest_path: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.is_file():
        archive.unlink()

    members: list[str] = [str(manifest_path.relative_to(REPO_ROOT))]
    for pair_id in pair_ids:
        members.append(str(paths.he_tiff(pair_id).relative_to(REPO_ROOT)))
        members.append(str(paths.ihc_tiff(pair_id).relative_to(REPO_ROOT)))

    cmd = ["tar", "-cf", "-", "-C", str(REPO_ROOT), *members]
    zstd = ["zstd", "-T0", "-3", "-o", str(archive)]
    print(f"packing {len(members)} members → {archive}", flush=True)
    p_tar = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p_zstd = subprocess.Popen(zstd, stdin=p_tar.stdout)
    assert p_tar.stdout is not None
    p_tar.stdout.close()
    z_code = p_zstd.wait()
    t_code = p_tar.wait()
    if t_code != 0 or z_code != 0:
        raise RuntimeError(f"tar|zstd failed (tar={t_code}, zstd={z_code})")


def main() -> None:
    if paths.SCALE != 1:
        raise SystemExit(f"paths.SCALE must be 1 for fair 1x pack; got {paths.SCALE}")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", type=int, help="pair ids (default: all)")
    ap.add_argument("--skip-export", action="store_true", help="do not re-export; require existing 1x tiffs")
    ap.add_argument("--force-export", action="store_true", help="always re-export even if 1x exists")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "regwsi" / "regwsi_inputs_1x.tar.zst",
    )
    args = ap.parse_args()

    n = crop_core.num_pairs()
    pair_ids = args.pairs if args.pairs is not None else list(range(n))
    for p in pair_ids:
        if p < 0 or p >= n:
            raise SystemExit(f"pair {p} out of range [0, {n})")

    excluded = sorted(p for p in pair_ids if p in paths.EXCLUDED_PAIR_IDS)
    if excluded:
        print(f"excluding corrupt L5 pairs: {excluded}", flush=True)
        pair_ids = [p for p in pair_ids if p not in paths.EXCLUDED_PAIR_IDS]

    failed: list[int] = list(excluded)
    if not args.skip_export:
        _, failed = export_all(pair_ids, force=args.force_export)
        pair_ids = [p for p in pair_ids if p not in failed]
    else:
        for pair_id in list(pair_ids):
            if not _is_scale1(paths.he_tiff(pair_id)) or not _is_scale1(paths.ihc_tiff(pair_id)):
                print(f"FAIL pair {pair_id}: missing SCALE=1 tiffs", flush=True)
                failed.append(pair_id)
        pair_ids = [p for p in pair_ids if p not in failed]

    if not pair_ids:
        raise SystemExit(f"no pairs to pack; failed={failed}")

    manifest_path = paths.REGWSI_ROOT / "pack_manifest_1x.json"
    paths.REGWSI_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(pair_ids, manifest_path)
    pack_tar_zst(pair_ids, args.output.resolve(), manifest_path)
    print(
        json.dumps(
            {
                "archive": str(args.output.resolve()),
                "archive_bytes": args.output.resolve().stat().st_size,
                "n_pairs": manifest["n_pairs"],
                "packed_pairs": pair_ids,
                "excluded_pairs": excluded,
                "failed_pairs": [p for p in failed if p not in paths.EXCLUDED_PAIR_IDS],
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
