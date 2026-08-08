"""Aggregate TRE matrix for an eval batch + pair."""

from __future__ import annotations

import json

from setup.coarse_to_fine import eval_runs, tre_eval
from setup.coarse_to_fine.identity import pair_fingerprint


def compute_batch_pair_tre(batch_id: str, pair_id: int) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        return {"error": f"no batch {batch_id}"}

    baseline = tre_eval.compute_pair_baseline(pair_id)
    if "error" in baseline:
        return baseline

    lams = list(manifest.get("lams") or [])
    estimators = list(manifest.get("estimators") or [])
    methods: list[dict] = []

    for lam in lams:
        for est in estimators:
            key = f"{lam}/{est}"
            path = eval_runs.tre_path(batch_id, pair_id, lam, est)
            cell = {
                "key": key,
                "lam": lam,
                "field_estimator": est,
                "complete": eval_runs.cell_complete(batch_id, pair_id, lam, est),
            }
            if path.is_file():
                try:
                    tre = json.loads(path.read_text())
                except Exception as e:
                    tre = tre_eval.empty_err(str(e))
                cell["tre"] = tre
            else:
                cell["tre"] = tre_eval.empty_err("not computed")
            methods.append(cell)

    return {
        "pair_id": pair_id,
        "batch_id": batch_id,
        "identity": pair_fingerprint(pair_id),
        "n": baseline["n"],
        "canvas": baseline["canvas"],
        "scale": baseline["scale"],
        "none": baseline["none"],
        "regwsi": baseline["regwsi"],
        "methods": methods,
        "lams": lams,
        "estimators": estimators,
    }
