"""Aggregate TRE matrix for an eval batch + pair."""

from __future__ import annotations

import json

from setup.coarse_to_fine import eval_runs, tre_eval
from setup.coarse_to_fine.identity import pair_fingerprint


def _runtime_from_cell(batch_id: str, pair_id: int, lam: str, est: str, tre: dict) -> float | None:
    if tre.get("runtime_s") is not None:
        try:
            return float(tre["runtime_s"])
        except (TypeError, ValueError):
            pass
    meta = eval_runs.read_cell_meta(batch_id, pair_id, lam, est)
    if meta and meta.get("runtime_s") is not None:
        try:
            return float(meta["runtime_s"])
        except (TypeError, ValueError):
            return None
    return None


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _batch_runtime_avgs(manifest: dict, batch_id: str) -> dict:
    pairs = [int(p) for p in manifest.get("pairs") or []]
    lams = list(manifest.get("lams") or [])
    estimators = list(manifest.get("estimators") or [])
    method_vals: dict[str, list[float]] = {
        f"{lam}/{est}": [] for lam in lams for est in estimators
    }
    regwsi_vals: list[float] = []

    for pair_id in pairs:
        rw = eval_runs.read_runtime_s(eval_runs.regwsi_runtime_path(batch_id, pair_id))
        if rw is not None:
            regwsi_vals.append(rw)
        for lam in lams:
            for est in estimators:
                key = f"{lam}/{est}"
                meta = eval_runs.read_cell_meta(batch_id, pair_id, lam, est)
                if not meta or meta.get("runtime_s") is None:
                    continue
                try:
                    method_vals[key].append(float(meta["runtime_s"]))
                except (TypeError, ValueError):
                    continue

    return {
        "regwsi": _mean(regwsi_vals),
        "methods": {k: _mean(v) for k, v in method_vals.items()},
    }


def compute_batch_pair_tre(batch_id: str, pair_id: int) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        return {"error": f"no batch {batch_id}"}

    baseline = tre_eval.compute_pair_baseline(pair_id)
    if "error" in baseline:
        return baseline

    cfg = manifest.get("config") or {}
    cell_cfg = eval_runs.cell_config(cfg)
    lams = list(manifest.get("lams") or [])
    estimators = list(manifest.get("estimators") or [])
    methods: list[dict] = []
    runtime_avgs = _batch_runtime_avgs(manifest, batch_id)

    for lam in lams:
        for est in estimators:
            key = f"{lam}/{est}"
            path = eval_runs.tre_path(batch_id, pair_id, lam, est)
            cell = {
                "key": key,
                "lam": lam,
                "field_estimator": est,
                "complete": eval_runs.cell_complete(
                    batch_id, pair_id, lam, est, cell_cfg
                ),
            }
            if path.is_file():
                try:
                    tre = json.loads(path.read_text())
                except Exception as e:
                    tre = tre_eval.empty_err(str(e))
                cell["tre"] = tre
            else:
                tre = tre_eval.empty_err("not computed")
                cell["tre"] = tre
            cell["runtime_s"] = _runtime_from_cell(batch_id, pair_id, lam, est, tre)
            cell["runtime_avg_s"] = runtime_avgs["methods"].get(key)
            methods.append(cell)

    regwsi_runtime = eval_runs.read_runtime_s(
        eval_runs.regwsi_runtime_path(batch_id, pair_id)
    )

    return {
        "pair_id": pair_id,
        "batch_id": batch_id,
        "identity": pair_fingerprint(pair_id),
        "n": baseline["n"],
        "canvas": baseline["canvas"],
        "scale": baseline["scale"],
        "none": baseline["none"],
        "regwsi": baseline["regwsi"],
        "regwsi_runtime_s": regwsi_runtime,
        "regwsi_runtime_avg_s": runtime_avgs["regwsi"],
        "methods": methods,
        "lams": lams,
        "estimators": estimators,
        "config": eval_runs.config_summary(cfg),
    }
