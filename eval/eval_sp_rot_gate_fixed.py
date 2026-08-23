"""
Re-score checkpoints with the corrected rigid gate (centre-fixed-point residual).

  python eval/eval_sp_rot_gate_fixed.py [ckpt.pt ...]

Prints per-angle n_matches / n_inliers / rot_err / centre residual for two-way NN
and LightGlue, and writes gate_fixed_eval.json next to the first checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "setup" / "live_crop"))
sys.path.insert(0, str(REPO / "introducing_superpoint"))

from setup import datasets
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_eval as beval
from training import DEFAULT_WEIGHTS

TILE = {"pair_id": 0, "x": 12, "y": 10, "loc": "12_10"}
ANGLES = list(range(0, 360, 30))
NN_THRESH = 0.7


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run(model, device, cfg) -> dict:
    return beval.evaluate_tile_matchers(
        model,
        [TILE],
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


def _report(name: str, ev: dict) -> None:
    for kind in ("nn", "lg"):
        cells = sorted((ev[kind].get("cells") or []), key=lambda c: float(c["angle"]))
        ok = [float(c["angle"]) for c in cells if c.get("auto_pass")]
        print(f"\n{name} / {kind}: pass {ev[kind]['n_pass']}/{ev[kind]['n_total']}  ok={ok}")
        print(f"{'ang':>5} {'match':>7} {'inlier':>7} {'rot_err':>9} {'centre_res':>11} {'pass':>6}")
        for c in cells:
            rot, tr = c.get("rot_err_deg"), c.get("trans_err_rel")
            print(
                f"{float(c['angle']):>5.0f} {str(c.get('n_matches')):>7} {str(c.get('n_inliers')):>7} "
                f"{'-' if rot is None else format(float(rot), '.3f'):>9} "
                f"{'-' if tr is None else format(float(tr), '.5f'):>11} "
                f"{str(bool(c.get('auto_pass'))):>6}"
            )


def main() -> None:
    datasets.set_active_dataset("muromi")
    cfg = {**store.DEFAULT_CONFIG}
    device = _device()
    print(f"device {device}  gate: rot<={beval.MAX_ROT_ERR_DEG} centre_res<={beval.MAX_TRANS_ERR_REL}")

    ckpts = [Path(a) for a in sys.argv[1:]]
    out = {}
    for label, path in [("magicleap_init", Path(DEFAULT_WEIGHTS))] + [
        (p.stem, p) for p in ckpts
    ]:
        model = beval.load_sp_model(path, device, nms_radius=int(cfg["sp_nms_dist"]))
        ev = _run(model, device, cfg)
        _report(label, ev)
        out[label] = {
            k: {
                "n_pass": ev[k]["n_pass"],
                "n_total": ev[k]["n_total"],
                "pass_rate": ev[k]["pass_rate"],
                "cells": ev[k]["cells"],
            }
            for k in ("nn", "lg")
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dest = (ckpts[0].parent if ckpts else REPO) / "gate_fixed_eval.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
