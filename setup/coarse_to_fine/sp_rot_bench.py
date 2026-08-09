"""SP+LightGlue rotation robustness bench store + GT compare helpers."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

import conf
from setup import datasets
from regWSI.extract_rigid import find_initial_df, rigid_from_displacement_field

SP_ROT_ROOT = conf.PROJECT_ROOT / "data" / "sp_rot_runs"
DEFAULT_ANGLES = list(range(0, 360, 30))
LABELS = ("pass", "fail", "unsure")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(name: str) -> str:
    s = _UNSAFE.sub("-", (name or "").strip()).strip("-").lower()
    return s or f"sp-rot-{int(time.time())}"


def run_dir(run_id: str) -> Path:
    return SP_ROT_ROOT / run_id


def manifest_path(run_id: str) -> Path:
    return run_dir(run_id) / "manifest.json"


def status_path(run_id: str) -> Path:
    return run_dir(run_id) / "status.json"


def labels_path(run_id: str) -> Path:
    return run_dir(run_id) / "labels.json"


def summary_path(run_id: str) -> Path:
    return run_dir(run_id) / "summary.json"


def pair_dir(run_id: str, pair_id: int) -> Path:
    return run_dir(run_id) / str(int(pair_id))


def cell_dir(run_id: str, pair_id: int, angle: int) -> Path:
    return pair_dir(run_id, pair_id) / str(int(angle))


def gt_copy_path(run_id: str, pair_id: int) -> Path:
    return pair_dir(run_id, pair_id) / "gt_rigid.json"


def label_key(pair_id: int, angle: int) -> str:
    return f"{int(pair_id)}:{int(angle)}"


def gt_rigid_store_path(pair_id: int, dataset: str | None = None) -> Path:
    ds = datasets.normalize_dataset(dataset) if dataset else datasets.active_dataset()
    if ds == "acrobat":
        return datasets.rigid_path(pair_id, ds)
    return conf.PROJECT_ROOT / "data" / "rigid" / "regwsi_initial" / f"{int(pair_id)}.json"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_manifest(run_id: str) -> dict | None:
    return _read_json(manifest_path(run_id))


def load_status(run_id: str) -> dict | None:
    return _read_json(status_path(run_id))


def write_status(run_id: str, status: dict) -> None:
    status_path(run_id).write_text(json.dumps(status, indent=2))


def load_labels(run_id: str) -> dict:
    raw = _read_json(labels_path(run_id))
    if not raw:
        return {}
    labels = raw.get("labels")
    return dict(labels) if isinstance(labels, dict) else {}


def save_label(
    run_id: str,
    pair_id: int,
    angle: int,
    label: str,
    note: str | None = None,
) -> dict:
    lab = (label or "").strip().lower()
    if lab not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    path = labels_path(run_id)
    store = _read_json(path) or {"run_id": run_id, "labels": {}}
    labels = store.setdefault("labels", {})
    entry: dict = {
        "pair_id": int(pair_id),
        "angle": int(angle),
        "label": lab,
        "labeled_at": int(time.time()),
    }
    if note is not None:
        entry["note"] = str(note)
    labels[label_key(pair_id, angle)] = entry
    store["updated_at"] = int(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    return entry


def list_runs() -> list[dict]:
    if not SP_ROT_ROOT.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(SP_ROT_ROOT.iterdir()):
        if not p.is_dir():
            continue
        man = _read_json(p / "manifest.json")
        if not man:
            continue
        st = _read_json(p / "status.json") or {}
        out.append(
            {
                "id": p.name,
                "name": man.get("name") or p.name,
                "dataset": man.get("dataset") or "muromi",
                "pairs": man.get("pairs") or [],
                "angles": man.get("angles") or DEFAULT_ANGLES,
                "created_at": man.get("created_at"),
                "status": st,
            }
        )
    out.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
    return out


def create_run(
    name: str,
    pairs: list[int],
    *,
    dataset: str = "muromi",
    run_id: str | None = None,
    angles: list[int] | None = None,
    preview_level: int = 2,
    notes: str | None = None,
) -> dict:
    ds = datasets.normalize_dataset(dataset)
    uniq: list[int] = []
    seen: set[int] = set()
    for p in pairs:
        pid = int(p)
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(pid)
    if not uniq:
        raise ValueError("no pairs")
    ang = [int(a) % 360 for a in (angles or DEFAULT_ANGLES)]
    ang = sorted(set(ang))
    rid = slugify(run_id or name)
    root = run_dir(rid)
    if root.exists():
        raise FileExistsError(f"run already exists: {rid}")
    root.mkdir(parents=True, exist_ok=True)
    total = len(uniq) * len(ang)
    man = {
        "id": rid,
        "name": name.strip() or rid,
        "dataset": ds,
        "pairs": uniq,
        "angles": ang,
        "preview_level": int(preview_level),
        "gt_source": "regwsi_initial",
        "notes": notes or "",
        "created_at": int(time.time()),
        "total_cells": total,
    }
    manifest_path(rid).write_text(json.dumps(man, indent=2))
    write_status(
        rid,
        {
            "state": "created",
            "done": 0,
            "total": total,
            "failed": 0,
            "skipped": 0,
            "detail": None,
            "error": None,
            "updated_at": int(time.time()),
        },
    )
    labels_path(rid).write_text(
        json.dumps({"run_id": rid, "labels": {}, "updated_at": int(time.time())}, indent=2)
    )
    return man


def _is_regwsi_gt(store: dict | None) -> bool:
    if not store:
        return False
    ver = str(store.get("version") or "")
    src = str(store.get("source") or "")
    return ver == "regwsi_initial" or src.startswith("regwsi")


def find_df_for_gt(pair_id: int, dataset: str | None = None) -> Path | None:
    out_dir = datasets.pair_dir(pair_id, dataset) / "out"
    pair_tmp_candidates = [
        datasets.pair_dir(pair_id, dataset) / "tmp",
        datasets.regwsi_root(dataset) / "tmp" / str(int(pair_id)),
        conf.PROJECT_ROOT / "data" / "regwsi" / "tmp" / str(int(pair_id)),
    ]
    for tmp in pair_tmp_candidates:
        found = find_initial_df(tmp, out_dir)
        if found is not None:
            return found
    found = find_initial_df(out_dir, out_dir)
    if found is not None:
        return found
    composed = out_dir / "displacement_field.mha"
    if composed.is_file():
        return composed
    return None


def ensure_gt_rigid(pair_id: int, dataset: str | None = None) -> dict:
    ds = datasets.normalize_dataset(dataset) if dataset else datasets.active_dataset()
    path = gt_rigid_store_path(pair_id, ds)
    existing = _read_json(path)
    if _is_regwsi_gt(existing) and existing.get("rigid"):
        return existing

    if ds == "acrobat":
        alt = datasets.rigid_path(pair_id, ds)
        alt_store = _read_json(alt)
        if _is_regwsi_gt(alt_store) and alt_store.get("rigid"):
            return alt_store

    df_path = find_df_for_gt(pair_id, ds)
    if df_path is None:
        raise FileNotFoundError(
            f"no regWSI displacement field for pair {pair_id} (dataset={ds})"
        )
    source = (
        "regwsi_initial"
        if "Initial_Registration" in df_path.parts
        else "regwsi_composed_fallback"
    )
    rigid_n, stats = rigid_from_displacement_field(df_path)
    store = {
        "pair_id": int(pair_id),
        "dataset": ds,
        "version": "regwsi_initial",
        "identity": datasets.pair_fingerprint(pair_id, ds),
        "rigid": rigid_n.tolist(),
        "stats": stats,
        "source": source,
        "df_path": str(df_path),
        "saved_at": int(time.time()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    return store


def _norm_rigid_to_pixel(M: np.ndarray, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(M, dtype=float).reshape(2, 3)
    sx, sy = float(w), float(h)
    R = np.array(
        [
            [M[0, 0], M[0, 1] * (sx / sy)],
            [M[1, 0] * (sy / sx), M[1, 1]],
        ],
        dtype=float,
    )
    t = np.array([M[0, 2] * sx, M[1, 2] * sy], dtype=float)
    return R, t


def rotation_deg_from_norm_rigid(M, w: float, h: float) -> float:
    R, _ = _norm_rigid_to_pixel(M, w, h)
    return float(np.degrees(np.arctan2(float(R[1, 0]), float(R[0, 0]))))


def angular_diff_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def compare_rigid_to_gt(
    pred_rigid,
    gt_rigid,
    *,
    width: float,
    height: float,
) -> dict:
    w, h = float(width), float(height)
    pred = np.asarray(pred_rigid, dtype=float).reshape(2, 3)
    gt = np.asarray(gt_rigid, dtype=float).reshape(2, 3)
    pred_deg = rotation_deg_from_norm_rigid(pred, w, h)
    gt_deg = rotation_deg_from_norm_rigid(gt, w, h)
    rot_err = abs(angular_diff_deg(pred_deg, gt_deg))

    _, t_pred = _norm_rigid_to_pixel(pred, w, h)
    _, t_gt = _norm_rigid_to_pixel(gt, w, h)
    trans_err_t = float(np.linalg.norm(t_pred - t_gt))

    pts_n = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    ones = np.ones((len(pts_n), 1), dtype=float)
    homo = np.concatenate([pts_n, ones], axis=1)
    pred_n = (pred @ homo.T).T
    gt_n = (gt @ homo.T).T
    d_px = (pred_n - gt_n) * np.array([w, h], dtype=float)
    center_err = float(np.linalg.norm(d_px[0]))
    mean_corner_err = float(np.mean(np.linalg.norm(d_px[1:], axis=1)))

    return {
        "pred_rotation_deg": pred_deg,
        "gt_rotation_deg": gt_deg,
        "rot_err_deg": rot_err,
        "trans_err_px": trans_err_t,
        "center_err_px": center_err,
        "mean_corner_err_px": mean_corner_err,
        "width": w,
        "height": h,
    }


def copy_run_artifacts(src_run: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "he.png",
        "ihc.png",
        "ihc_prerot.png",
        "ihc_rigid.png",
        "matches.png",
        "matches.json",
        "result.json",
    ):
        src = src_run / name
        if src.is_file():
            shutil.copy2(src, dest / name)


def cell_result_path(run_id: str, pair_id: int, angle: int) -> Path:
    return cell_dir(run_id, pair_id, angle) / "result.json"


def load_cell_result(run_id: str, pair_id: int, angle: int) -> dict | None:
    return _read_json(cell_result_path(run_id, pair_id, angle))


def build_summary(run_id: str) -> dict:
    man = load_manifest(run_id)
    if not man:
        raise FileNotFoundError(f"no manifest for {run_id}")
    pairs = [int(p) for p in man.get("pairs") or []]
    angles = [int(a) for a in man.get("angles") or DEFAULT_ANGLES]
    labels = load_labels(run_id)

    by_angle: dict[str, dict] = {}
    metric_by_label: dict[str, dict[str, list[float]]] = {
        lab: {"n_inliers": [], "rmse_px": [], "rot_err_deg": [], "trans_err_px": []}
        for lab in LABELS
    }
    cells_done = 0
    cells_failed = 0

    for ang in angles:
        counts = {lab: 0 for lab in LABELS}
        counts["unlabeled"] = 0
        metric_lists = {
            "n_inliers": [],
            "rmse_px": [],
            "rot_err_deg": [],
            "trans_err_px": [],
        }
        for pid in pairs:
            res = load_cell_result(run_id, pid, ang)
            key = label_key(pid, ang)
            lab_entry = labels.get(key)
            lab = None
            if isinstance(lab_entry, dict):
                lab = str(lab_entry.get("label") or "").lower() or None
            if lab in LABELS:
                counts[lab] += 1
            else:
                counts["unlabeled"] += 1

            if not res:
                continue
            if res.get("error"):
                cells_failed += 1
                continue
            cells_done += 1
            n_in = res.get("n_inliers")
            rmse = (res.get("stats") or {}).get("rmse_px")
            rot = res.get("rot_err_deg")
            tr = res.get("trans_err_px")
            for name, val in (
                ("n_inliers", n_in),
                ("rmse_px", rmse),
                ("rot_err_deg", rot),
                ("trans_err_px", tr),
            ):
                if val is None:
                    continue
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    continue
                metric_lists[name].append(f)
                if lab in LABELS:
                    metric_by_label[lab][name].append(f)

        labeled = counts["pass"] + counts["fail"] + counts["unsure"]
        fail_rate = (counts["fail"] / labeled) if labeled else None
        by_angle[str(ang)] = {
            "counts": counts,
            "labeled": labeled,
            "fail_rate": fail_rate,
            "metrics": {
                k: {
                    "n": len(v),
                    "mean": float(np.mean(v)) if v else None,
                    "median": float(np.median(v)) if v else None,
                }
                for k, v in metric_lists.items()
            },
        }

    summary = {
        "run_id": run_id,
        "dataset": man.get("dataset"),
        "pairs": pairs,
        "angles": angles,
        "cells_with_result": cells_done,
        "cells_failed": cells_failed,
        "by_angle": by_angle,
        "metrics_by_label": {
            lab: {
                k: {
                    "n": len(v),
                    "mean": float(np.mean(v)) if v else None,
                    "median": float(np.median(v)) if v else None,
                }
                for k, v in mets.items()
            }
            for lab, mets in metric_by_label.items()
        },
        "built_at": int(time.time()),
    }
    summary_path(run_id).write_text(json.dumps(summary, indent=2))
    return summary


def matrix_status(run_id: str) -> dict:
    man = load_manifest(run_id)
    if not man:
        raise FileNotFoundError(f"no manifest for {run_id}")
    pairs = [int(p) for p in man.get("pairs") or []]
    angles = [int(a) for a in man.get("angles") or DEFAULT_ANGLES]
    labels = load_labels(run_id)
    cells: list[dict] = []
    for pid in pairs:
        for ang in angles:
            res = load_cell_result(run_id, pid, ang)
            key = label_key(pid, ang)
            lab_entry = labels.get(key)
            lab = None
            if isinstance(lab_entry, dict):
                lab = lab_entry.get("label")
            entry = {
                "pair_id": pid,
                "angle": ang,
                "state": "missing",
                "label": lab,
                "n_inliers": None,
                "rmse_px": None,
                "rot_err_deg": None,
                "trans_err_px": None,
                "error": None,
            }
            if res:
                if res.get("error"):
                    entry["state"] = "error"
                    entry["error"] = res.get("error")
                else:
                    entry["state"] = "done"
                    entry["n_inliers"] = res.get("n_inliers")
                    entry["rmse_px"] = (res.get("stats") or {}).get("rmse_px")
                    entry["rot_err_deg"] = res.get("rot_err_deg")
                    entry["trans_err_px"] = res.get("trans_err_px")
            cells.append(entry)
    return {
        "run_id": run_id,
        "manifest": man,
        "status": load_status(run_id) or {},
        "cells": cells,
    }
