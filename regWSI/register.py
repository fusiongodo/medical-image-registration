"""
Run DeeperHistReg (regWSI) affine + deformable preregistration for one pair.

Uses default_initial_nonrigid(); the composed transform is saved as
data/regwsi/{pair}/out/displacement_field.mha ("the sum"). The registration's
warped source is copied to out/warped_ihc.tiff.

Usage:
  python regWSI/register.py <pair_id>
  python regWSI/register.py <pair_id> --export   # export RGB inputs first
  python regWSI/register.py <pair_id> --preview  # also build preview PNGs
  python regWSI/register.py <pair_id> --full     # also build 2x2 full-res explorer JPEGs
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from setup.coarse_to_fine.identity import pair_fingerprint

from regWSI import paths
from regWSI.export_slides import export_pair


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _patch_device(params: dict, device: str) -> None:
    params["device"] = device
    init = params.get("initial_registration_params") or {}
    init["device"] = device
    init["cuda"] = device != "cpu"
    params["initial_registration_params"] = init
    nonrigid = params.get("nonrigid_registration_params") or {}
    nonrigid["device"] = device
    params["nonrigid_registration_params"] = nonrigid


def _find_warped_source(out: Path) -> Path | None:
    for p in out.iterdir():
        if p.is_file() and "warped_source" in p.name:
            return p
    return None


def register_pair(pair_id: int, do_export: bool = False, force_export: bool = False) -> dict:
    if do_export or not paths.he_tiff(pair_id).is_file() or not paths.ihc_tiff(pair_id).is_file():
        export_pair(pair_id, force=force_export)

    he = paths.he_tiff(pair_id)
    ihc = paths.ihc_tiff(pair_id)
    if not he.is_file() or not ihc.is_file():
        raise FileNotFoundError(f"missing RGB inputs for pair {pair_id}; run export_slides first")

    paths.ensure_pair_dirs(pair_id)
    out = paths.out_dir(pair_id)

    import deeperhistreg

    registration_params = deeperhistreg.configs.default_initial_nonrigid()
    device = _device()
    _patch_device(registration_params, device)
    registration_params["loading_params"]["loader"] = "tiff"

    temp = paths.pair_dir(pair_id) / "tmp"
    if temp.exists():
        shutil.rmtree(temp)

    config = {
        "source_path": str(ihc),
        "target_path": str(he),
        "output_path": str(out),
        "registration_parameters": registration_params,
        "case_name": f"pair_{pair_id}",
        "save_displacement_field": True,
        "copy_target": True,
        "delete_temporary_results": True,
        "temporary_path": str(temp),
    }
    deeperhistreg.run_registration(**config)

    df = paths.displacement_field(pair_id)
    if not df.is_file():
        raise RuntimeError(
            f"registration finished without displacement_field.mha for pair {pair_id}; "
            f"check logs under {out}"
        )

    warped = _find_warped_source(out)
    dest = paths.warped_ihc(pair_id)
    if warped is not None:
        if warped.resolve() != dest.resolve():
            shutil.copy2(warped, dest)
    else:
        from deeperhistreg.dhr_input_output.dhr_savers import tiff_saver as _tiff_saver

        deeperhistreg.apply_deformation(
            source_image_path=str(ihc),
            target_image_path=str(he),
            warped_image_path=str(dest),
            displacement_field_path=str(df),
            loader=deeperhistreg.loaders.TIFFLoader,
            saver=deeperhistreg.savers.TIFFSaver,
            save_params=_tiff_saver.default_params,
            level=0,
            pad_value=255.0,
            save_source_only=True,
            to_template_shape=True,
        )

    meta = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "level": paths.LEVEL,
        "scale": paths.SCALE,
        "canvas": [paths.CANVAS_W, paths.CANVAS_H],
        "params": "default_initial_nonrigid",
        "device": device,
        "source": "ihc",
        "target": "he",
        "displacement_field": str(df.relative_to(REPO_ROOT)),
        "warped_ihc": str(dest.relative_to(REPO_ROOT)) if dest.is_file() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    paths.meta_json(pair_id).write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", type=int, help="pair index")
    ap.add_argument("--export", action="store_true", help="(re)export RGB inputs first")
    ap.add_argument("--force-export", action="store_true", help="overwrite existing RGB tiffs")
    ap.add_argument("--preview", action="store_true", help="build preview PNGs after register")
    ap.add_argument("--full", action="store_true", help="build 2x2 full-res explorer JPEGs after register")
    args = ap.parse_args()
    meta = register_pair(args.pair, do_export=args.export or args.force_export, force_export=args.force_export)
    print(json.dumps(meta, indent=2))
    if args.preview:
        from regWSI.make_preview import make_preview

        print(json.dumps(make_preview(args.pair), indent=2))
    if args.full:
        from regWSI.make_full import make_full

        print(json.dumps(make_full(args.pair), indent=2))


if __name__ == "__main__":
    main()
