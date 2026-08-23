"""
Final evaluation of the random-angle SuperPoint fine-tune, stock vs tuned.

  python eval/eval_sp_rot_final.py --mode within
  python eval/eval_sp_rot_final.py --mode cross

within: a held-out H&E crop against a rotated copy of itself, so the transform is
        exactly rigid and no registration error enters. Measures rotation invariance
        in isolation.
cross:  the raw H&E slide against the raw IHC slide pre-rotated by a known angle. The
        matcher has to discover the H&E->IHC rigid transform, which is then composed
        with the known pre-rotation and scored against the regWSI ground truth. Needs
        no pre-registration because the alignment is the output, not an input.

Both modes match with two-way NN and LightGlue off one shared extract. Results go to
eval/out/sp_rot_final_<mode>.json as one record per cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "setup" / "live_crop"))
sys.path.insert(0, str(REPO / "introducing_superpoint"))

from setup import datasets
from setup.coarse_to_fine import rigid_sp_lg
from setup.coarse_to_fine import sp_rot_bench as bench
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_eval as beval
from training import DEFAULT_WEIGHTS

RUN_ID = "rot_rand_d4"
ANGLES = [float(a) for a in range(0, 360, 30)]
NN_THRESH = 0.7
CROSS_RESIZES = [512, 1024, 2048]
OUT_DIR = REPO / "eval" / "out"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _weight_paths(cfg: dict) -> list[tuple[str, Path]]:
    tuned = store.ckpt_dir(RUN_ID) / "latest.pt"
    if not tuned.is_file():
        raise FileNotFoundError(f"no tuned checkpoint at {tuned}")
    return [("stock", Path(DEFAULT_WEIGHTS)), ("tuned", tuned)]


def _summarise(cells: list[dict]) -> dict:
    n_pass = sum(1 for c in cells if c.get("auto_pass"))
    by_angle: dict[str, dict] = {}
    for c in cells:
        a = f"{float(c['angle']):g}"
        d = by_angle.setdefault(a, {"n": 0, "n_pass": 0, "matches": [], "inliers": [], "rot": []})
        d["n"] += 1
        d["n_pass"] += 1 if c.get("auto_pass") else 0
        d["matches"].append(int(c.get("n_matches") or 0))
        d["inliers"].append(int(c.get("n_inliers") or 0))
        if c.get("rot_err_deg") is not None:
            d["rot"].append(float(c["rot_err_deg"]))
    for d in by_angle.values():
        d["pass_rate"] = d["n_pass"] / d["n"] if d["n"] else None
        d["matches_mean"] = float(np.mean(d["matches"])) if d["matches"] else None
        d["inliers_mean"] = float(np.mean(d["inliers"])) if d["inliers"] else None
        d["rot_err_median"] = float(np.median(d["rot"])) if d["rot"] else None
        d["rot_err_max"] = float(np.max(d["rot"])) if d["rot"] else None
        for k in ("matches", "inliers", "rot"):
            d.pop(k)
    return {
        "n_pass": n_pass,
        "n_total": len(cells),
        "pass_rate": (n_pass / len(cells)) if cells else None,
        "by_angle": by_angle,
        "cells": cells,
    }


def run_within(cfg: dict, device: torch.device, n_tiles: int, split_name: str) -> dict:
    split = store.load_split(RUN_ID)
    tiles = list(split[split_name])[:n_tiles]
    out = {
        "mode": "within",
        "split": split_name,
        "angles": ANGLES,
        "nn_thresh": NN_THRESH,
        "extract_resize": int(cfg["extract_resize"]),
        "sp_nms_dist": int(cfg["sp_nms_dist"]),
        "n_tiles": len(tiles),
        "tiles": [{"pair_id": int(t["pair_id"]), "loc": str(t.get("loc"))} for t in tiles],
        "gate": {
            "max_rot_err_deg": beval.MAX_ROT_ERR_DEG,
            "max_trans_err_rel": beval.MAX_TRANS_ERR_REL,
        },
        "results": {},
    }
    for label, path in _weight_paths(cfg):
        model = beval.load_sp_model(path, device, nms_radius=int(cfg["sp_nms_dist"]))
        ev = beval.evaluate_tile_matchers(
            model,
            tiles,
            angles=ANGLES,
            device=device,
            extract_resize=int(cfg["extract_resize"]),
            nms=int(cfg["sp_nms_dist"]),
            depth=int(cfg["depth"]),
            preview_level=int(cfg["preview_level"]),
            src_size=int(cfg["src_size"]),
            out_size=int(cfg["out_size"]),
            dataset=str(cfg.get("dataset") or "muromi"),
            nn_thresh=NN_THRESH,
        )
        out["results"][label] = {k: _summarise(ev[k]["cells"]) for k in ("nn", "lg")}
        for k in ("nn", "lg"):
            s = out["results"][label][k]
            print(
                f"within[{split_name}] {label:>5} {k}: pass {s['n_pass']}/{s['n_total']}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _gt_pairs() -> list[int]:
    usable = []
    for p in range(datasets.pair_count()):
        st = bench._read_json(bench.gt_rigid_store_path(p, "muromi"))
        if bench._is_regwsi_gt(st) and st.get("rigid"):
            usable.append(p)
        elif bench.find_df_for_gt(p, "muromi") is not None:
            usable.append(p)
    return usable


def _cross_cell(model, lg, device, hp, he, ihc, angle, gt, resize, inlier_px) -> dict:
    ihc_pre, pre_M = rigid_sp_lg._rotate_gray(ihc, float(angle))
    f0, (w, h) = beval.extract_feats(model, he, device, resize)
    f1, _ = beval.extract_feats(model, ihc_pre, device, resize)
    n_kp = (int(f0["keypoints"].shape[1]), int(f1["keypoints"].shape[1]))
    cells = {}
    for kind in ("nn", "lg"):
        try:
            if kind == "nn":
                q0, q1, _ = beval.match_nn_two_way(f0, f1, NN_THRESH)
            else:
                q0, q1, _ = beval.match_lg(f0, f1, device, hp, matcher=lg)
            n_match = int(len(q0))
            if n_match < 3:
                raise RuntimeError(f"only {n_match} matches")
            R, t, _mask, st = rigid_sp_lg.fit_rigid_kabsch(q0, q1, inlier_px)
            rigid_final = rigid_sp_lg._compose_norm_rigid(pre_M, R, t, w, h)
            cm = bench.compare_rigid_to_gt(
                rigid_final, gt.get("rigid"), width=float(w), height=float(h)
            )
            rot = cm.get("rot_err_deg")
            tr = cm.get("trans_err_px")
            trl = float(tr) / min(float(w), float(h)) if tr is not None else None
            ok = (
                rot is not None
                and trl is not None
                and float(rot) <= beval.MAX_ROT_ERR_DEG
                and trl <= beval.MAX_TRANS_ERR_REL
            )
            cell = {
                "n_matches": n_match,
                "n_inliers": int(st.get("n_inliers") or 0),
                "rot_err_deg": float(rot) if rot is not None else None,
                "trans_err_rel": trl,
                "auto_pass": bool(ok),
                "error": None,
            }
        except Exception as e:
            cell = {
                "n_matches": 0,
                "n_inliers": 0,
                "rot_err_deg": None,
                "trans_err_rel": None,
                "auto_pass": False,
                "error": str(e),
            }
        cell.update({"angle": float(angle), "n_kp0": n_kp[0], "n_kp1": n_kp[1]})
        cells[kind] = cell
    return cells


def run_cross(cfg: dict, device: torch.device, resizes: list[int]) -> dict:
    import crop_core

    pairs = _gt_pairs()
    hp = {**rigid_sp_lg.DEFAULT_HYPERPARAMS, "sp_nms_dist": int(cfg["sp_nms_dist"])}
    inlier_px = float(hp.get("rigid_inlier_px", 3.0))
    pages, gts = {}, {}
    for p in pairs:
        he = crop_core.whole_gray(p, "he", int(cfg["preview_level"]))
        ihc = crop_core.whole_gray(p, "ihc", int(cfg["preview_level"]))
        if he is None or ihc is None:
            continue
        pages[p] = (he, ihc)
        gts[p] = bench.ensure_gt_rigid(p, "muromi")
    pairs = sorted(pages)
    print(f"cross: {len(pairs)} pairs with regWSI GT: {pairs}", flush=True)

    out = {
        "mode": "cross",
        "pairs": pairs,
        "angles": ANGLES,
        "nn_thresh": NN_THRESH,
        "resizes": list(resizes),
        "preview_level": int(cfg["preview_level"]),
        "sp_nms_dist": int(cfg["sp_nms_dist"]),
        "max_num_keypoints": 2048,
        "page_shape": list(pages[pairs[0]][0].shape[:2]),
        "gate": {
            "max_rot_err_deg": beval.MAX_ROT_ERR_DEG,
            "max_trans_err_rel": beval.MAX_TRANS_ERR_REL,
        },
        "results": {},
    }
    for label, path in _weight_paths(cfg):
        model = beval.load_sp_model(path, device, nms_radius=int(cfg["sp_nms_dist"]))
        lg = beval.build_lg_matcher(device, hp)
        out["results"][label] = {}
        for rz in resizes:
            acc = {"nn": [], "lg": []}
            for p in pairs:
                he, ihc = pages[p]
                for a in ANGLES:
                    cells = _cross_cell(model, lg, device, hp, he, ihc, a, gts[p], rz, inlier_px)
                    for kind, cell in cells.items():
                        cell["pair_id"] = int(p)
                        acc[kind].append(cell)
            out["results"][label][str(rz)] = {k: _summarise(acc[k]) for k in ("nn", "lg")}
            for k in ("nn", "lg"):
                s = out["results"][label][str(rz)][k]
                print(
                    f"cross {label:>5} rz={rz:>4} {k}: pass {s['n_pass']}/{s['n_total']}",
                    flush=True,
                )
        del model, lg
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("within", "cross"))
    ap.add_argument("--tiles", type=int, default=12, help="held-out tiles for within mode")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--resizes", default=",".join(str(r) for r in CROSS_RESIZES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets.set_active_dataset("muromi")
    cfg = store.load_config(RUN_ID)
    device = _device()
    print(
        f"mode={args.mode} device={device} "
        f"gate: rot<={beval.MAX_ROT_ERR_DEG} centre_res<={beval.MAX_TRANS_ERR_REL}",
        flush=True,
    )

    if args.mode == "within":
        payload = run_within(cfg, device, int(args.tiles), str(args.split))
        stem = f"within_{args.split}"
    else:
        resizes = [int(x) for x in args.resizes.split(",") if x.strip()]
        payload = run_cross(cfg, device, resizes)
        stem = "cross"

    payload["run_id"] = RUN_ID
    payload["weights"] = {k: str(v) for k, v in _weight_paths(cfg)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = Path(args.out) if args.out else OUT_DIR / f"sp_rot_final_{stem}.json"
    dest.write_text(json.dumps(payload, indent=2))
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
