"""Fit pass/fail margins from human SP-rotation labels.

Primary gate (recommended): GT rotation + relative translation error
    pass ⇔ rot_err_deg ≤ T_rot AND trans_err_px / min(W,H) ≤ T_trans

Secondary: deployable features available without GT (n_inliers, rmse_px, …).

Usage:
  .venv/bin/python3 setup/coarse_to_fine/sp_rot_label_metrics.py sp-rot-1786287542465
  .venv/bin/python3 setup/coarse_to_fine/sp_rot_label_metrics.py sp-rot-1786287542465 --out /tmp/gt_gate.json
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup.coarse_to_fine import sp_rot_bench as bench


def _f(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def load_rows(run_id: str) -> list[dict]:
    man = bench.load_manifest(run_id)
    if not man:
        raise FileNotFoundError(f"no manifest for {run_id}")
    labels = bench.load_labels(run_id)
    rows = []
    for pid in man.get("pairs") or []:
        for ang in man.get("angles") or bench.DEFAULT_ANGLES:
            key = bench.label_key(pid, ang)
            lab_entry = labels.get(key) or {}
            lab = str(lab_entry.get("label") or "").lower() or None
            res = bench.load_cell_result(run_id, int(pid), int(ang)) or {}
            stats = res.get("stats") or {}
            w = _f(res.get("width")) or _f(stats.get("width"))
            h = _f(res.get("height")) or _f(stats.get("height"))
            tr_px = _f(res.get("trans_err_px"))
            trans_rel = None
            if tr_px is not None and w is not None and h is not None and min(w, h) > 0:
                trans_rel = tr_px / min(w, h)
            rows.append(
                {
                    "pair": int(pid),
                    "angle": int(ang),
                    "label": lab if lab in bench.LABELS else None,
                    "error": res.get("error"),
                    "n_inliers": _f(res.get("n_inliers")),
                    "n_matches": _f(res.get("n_matches")),
                    "rmse_px": _f(stats.get("rmse_px")),
                    "rot_err_deg": _f(res.get("rot_err_deg")),
                    "trans_err_px": tr_px,
                    "trans_err_rel": trans_rel,
                    "width": w,
                    "height": h,
                }
            )
    return rows


def summarize(vals: list[float]) -> dict | None:
    if not vals:
        return None
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal = 0.5 * (rec + spec)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "balanced_acc": bal,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "disagreements": fp + fn,
    }


def sweep_ge(x: np.ndarray, y: np.ndarray, name: str) -> list[dict]:
    vals = sorted({float(v) for v in x if np.isfinite(v)})
    out = []
    for thr in vals:
        pred = (x >= thr).astype(int)
        m = confusion(y, pred)
        out.append({"feature": name, "op": ">=", "threshold": thr, **m})
    return out


def sweep_le(x: np.ndarray, y: np.ndarray, name: str) -> list[dict]:
    vals = sorted({float(v) for v in x if np.isfinite(v)})
    out = []
    for thr in vals:
        pred = (x <= thr).astype(int)
        m = confusion(y, pred)
        out.append({"feature": name, "op": "<=", "threshold": thr, **m})
    return out


def best(candidates: list[dict], key: str = "balanced_acc") -> dict | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: (
            d.get(key, 0),
            d.get("f1", 0),
            d.get("precision", 0),
            -d.get("disagreements", 10**9),
        ),
    )


def _metric_slice(d: dict) -> dict:
    keys = (
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_acc",
        "accuracy",
        "disagreements",
    )
    return {k: d[k] for k in keys if k in d}


def analyze(run_id: str) -> dict:
    rows = load_rows(run_id)
    labeled = [r for r in rows if r["label"] in ("pass", "fail")]
    unsure = sum(1 for r in rows if r["label"] == "unsure")
    hard_err = [r for r in labeled if r.get("error")]
    usable_gt = [
        r
        for r in labeled
        if not r.get("error")
        and r["rot_err_deg"] is not None
        and r["trans_err_rel"] is not None
    ]
    usable_dep = [r for r in labeled if not r.get("error") and r["n_inliers"] is not None]

    y_gt = np.array([1 if r["label"] == "pass" else 0 for r in usable_gt], dtype=int)
    rot = np.array([r["rot_err_deg"] for r in usable_gt], dtype=float)
    tr_px = np.array([r["trans_err_px"] for r in usable_gt], dtype=float)
    tr_rel = np.array([r["trans_err_rel"] for r in usable_gt], dtype=float)

    by_lab = {}
    for lab in ("pass", "fail"):
        xs = [r for r in usable_gt if r["label"] == lab]
        by_lab[lab] = {
            "n": len(xs),
            "rot_err_deg": summarize([r["rot_err_deg"] for r in xs]),
            "trans_err_px": summarize([r["trans_err_px"] for r in xs]),
            "trans_err_rel": summarize([r["trans_err_rel"] for r in xs]),
            "n_inliers": summarize(
                [r["n_inliers"] for r in xs if r["n_inliers"] is not None]
            ),
            "rmse_px": summarize([r["rmse_px"] for r in xs if r["rmse_px"] is not None]),
        }

    gt_joint = []
    rot_vals = sorted({float(v) for v in rot})
    tr_vals = sorted({float(v) for v in tr_rel})
    for tr_thr in rot_vals:
        for tt in tr_vals:
            pred = ((rot <= tr_thr) & (tr_rel <= tt)).astype(int)
            m = confusion(y_gt, pred)
            gt_joint.append(
                {
                    "feature": "rot_err_deg<=T & trans_err_rel<=U",
                    "op": "and",
                    "max_rot_err_deg": tr_thr,
                    "max_trans_err_rel": tt,
                    **m,
                }
            )
    best_gt_joint = best(gt_joint)
    zero_disagree = [c for c in gt_joint if c.get("disagreements", 1) == 0]
    best_zero = None
    if zero_disagree:
        best_zero = min(
            zero_disagree,
            key=lambda d: (d["max_rot_err_deg"], d["max_trans_err_rel"]),
        )

    oracle = []
    if len(rot):
        oracle.extend(sweep_le(rot, y_gt, "rot_err_deg"))
    if len(tr_rel):
        oracle.extend(sweep_le(tr_rel, y_gt, "trans_err_rel"))
    if len(tr_px):
        oracle.extend(sweep_le(tr_px, y_gt, "trans_err_px"))
    best_oracle = best(oracle)

    recommended = None
    pick = best_zero or best_gt_joint
    if pick:
        recommended = {
            "kind": "gt_rot_and_trans_rel",
            "max_rot_err_deg": pick["max_rot_err_deg"],
            "max_trans_err_rel": pick["max_trans_err_rel"],
            "predict": (
                "pass if rot_err_deg <= max_rot_err_deg "
                "and trans_err_px / min(W,H) <= max_trans_err_rel"
            ),
            "metrics": _metric_slice(pick),
            "zero_disagreement": pick.get("disagreements") == 0,
        }

    y_dep = np.array([1 if r["label"] == "pass" else 0 for r in usable_dep], dtype=int)
    inl = np.array([r["n_inliers"] for r in usable_dep], dtype=float)
    rmse = np.array(
        [r["rmse_px"] if r["rmse_px"] is not None else np.nan for r in usable_dep],
        dtype=float,
    )
    nmat = np.array(
        [r["n_matches"] if r["n_matches"] is not None else np.nan for r in usable_dep],
        dtype=float,
    )

    cand = []
    if len(inl):
        cand.extend(sweep_ge(inl, y_dep, "n_inliers"))
    finite_rmse = np.isfinite(rmse)
    if finite_rmse.any():
        cand.extend(sweep_le(rmse[finite_rmse], y_dep[finite_rmse], "rmse_px"))
    finite_nmat = np.isfinite(nmat)
    if finite_nmat.any():
        cand.extend(sweep_ge(nmat[finite_nmat], y_dep[finite_nmat], "n_matches"))

    combo = []
    inl_vals = sorted({float(v) for v in inl}) if len(inl) else []
    rmse_vals = sorted({float(v) for v in rmse if np.isfinite(v)})
    for ti in inl_vals:
        for trm in rmse_vals:
            pred = ((inl >= ti) & np.isfinite(rmse) & (rmse <= trm)).astype(int)
            m = confusion(y_dep, pred)
            combo.append(
                {
                    "feature": "n_inliers>=T & rmse_px<=R",
                    "op": "and",
                    "threshold_inliers": ti,
                    "threshold_rmse": trm,
                    **m,
                }
            )

    best_single = best(cand)
    best_combo = best(combo)
    dep_gate = best_combo or best_single
    deployable = None
    if dep_gate:
        if dep_gate.get("feature") == "n_inliers>=T & rmse_px<=R":
            deployable = {
                "kind": "inliers_and_rmse",
                "min_inliers": dep_gate["threshold_inliers"],
                "max_rmse_px": dep_gate["threshold_rmse"],
                "predict": "pass if n_inliers >= min_inliers and rmse_px <= max_rmse_px",
                "metrics": _metric_slice(dep_gate),
            }
        else:
            deployable = {
                "kind": "single",
                "feature": dep_gate["feature"],
                "op": dep_gate["op"],
                "threshold": dep_gate["threshold"],
                "predict": f"pass if {dep_gate['feature']} {dep_gate['op']} {dep_gate['threshold']}",
                "metrics": _metric_slice(dep_gate),
            }

    by_angle = {}
    for ang in sorted({r["angle"] for r in usable_gt}):
        xs = [r for r in usable_gt if r["angle"] == ang]
        n_pass = sum(1 for r in xs if r["label"] == "pass")
        by_angle[str(ang)] = {
            "n": len(xs),
            "n_pass": n_pass,
            "pass_rate": n_pass / len(xs) if xs else None,
        }

    return {
        "run_id": run_id,
        "n_cells": len(rows),
        "n_labeled_pass_fail": len(labeled),
        "n_unsure": unsure,
        "n_hard_errors": len(hard_err),
        "n_usable_gt": len(usable_gt),
        "n_usable_deployable": len(usable_dep),
        "label_counts": dict(Counter(r["label"] for r in labeled)),
        "by_label": by_lab,
        "by_angle": by_angle,
        "recommended_gate": recommended,
        "tightest_zero_disagreement_gate": (
            {
                "max_rot_err_deg": best_zero["max_rot_err_deg"],
                "max_trans_err_rel": best_zero["max_trans_err_rel"],
                "metrics": _metric_slice(best_zero),
            }
            if best_zero
            else None
        ),
        "best_single_gt_feature": best_oracle,
        "deployable_gate": deployable,
        "best_single_deployable_feature": best_single,
        "best_deployable_combo": best_combo,
        "note": (
            "recommended_gate is GT rot∧trans_rel vs human labels. "
            "deployable_gate is n_inliers/rmse without GT."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = analyze(args.run_id)
    text = bench.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
