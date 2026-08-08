"""
Compute TRE for regWSI vs curated FFT field-set branches (TPS / Wendland / B-spline).

Landmarks in data/regwsi/{pair}/landmarks.json use normalised [0,1] HE/IHC coords.
Prints one JSON object to stdout.

Usage:
  python regWSI/tre_cli.py <pair_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import conf
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.reg_branches import FIELD_ESTIMATORS
from setup.coarse_to_fine import tre_eval

LAM = "fft"

# Back-compat aliases for make_fieldset_full / overlays
GRID = tre_eval.GRID


def _resolve_set_dir(pair_id: int, field_estimator: str):
    return tre_eval.resolve_curated_set_dir(pair_id, LAM, field_estimator)


def _branch_tre(
    points: list[dict],
    pair_id: int,
    field_estimator: str,
    w: int,
    h: int,
    scale: float,
) -> dict:
    set_id, set_dir, set_name = tre_eval.resolve_curated_set_dir(
        pair_id, LAM, field_estimator
    )
    meta = {
        "field_set_id": set_id,
        "field_set_name": set_name,
        "field_estimator": field_estimator,
    }
    if set_dir is None:
        out = tre_eval.empty_err(
            f"no field.json on {LAM}/{field_estimator}"
            + (f" (main/active={set_id})" if set_id else "")
        )
        out.update(meta)
        return out
    st = tre_eval.stats(
        tre_eval.tre_field_file(points, set_dir / "field.json", w, h, scale)
    )
    st.update(meta)
    return st


def compute_tre(pair_id: int) -> dict:
    points = tre_eval.load_landmarks(pair_id)
    w, h, scale = tre_eval.canvas_scale(pair_id)

    result: dict = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "n": len(points),
        "canvas": [w, h],
        "scale": scale,
        "tile_w": conf.CNN_INPUT_WIDTH,
        "tile_h": conf.CNN_INPUT_HEIGHT,
    }

    if not points:
        empty = tre_eval.stats(np.array([]))
        result["none"] = empty
        result["regwsi"] = empty
        result["tps"] = empty
        result["wendland"] = empty
        result["ours"] = empty
        result["field_set_id"] = None
        return result

    result["none"] = tre_eval.annotate_tile_means(
        tre_eval.stats(tre_eval.tre_none(points, w, h)), scale
    )
    try:
        result["regwsi"] = tre_eval.annotate_tile_means(
            tre_eval.stats(tre_eval.tre_regwsi(points, pair_id, w, h)), scale
        )
    except Exception as e:
        result["regwsi"] = tre_eval.empty_err(str(e))

    for est in FIELD_ESTIMATORS:
        result[est] = tre_eval.annotate_tile_means(
            _branch_tre(points, pair_id, est, w, h, scale), scale
        )

    if result["tps"].get("mean") is not None:
        result["ours"] = {**result["tps"]}
        result["field_set_id"] = result["tps"].get("field_set_id")
    elif result["wendland"].get("mean") is not None:
        result["ours"] = {**result["wendland"]}
        result["field_set_id"] = result["wendland"].get("field_set_id")
    else:
        result["ours"] = tre_eval.empty_err(
            result["tps"].get("error")
            or result["wendland"].get("error")
            or "no main field set"
        )
        result["field_set_id"] = None

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pair", type=int)
    args = ap.parse_args()
    print(json.dumps(compute_tre(args.pair), indent=2))


if __name__ == "__main__":
    main()
