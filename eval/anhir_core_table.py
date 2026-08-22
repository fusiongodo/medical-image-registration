"""
CORE-style rTRE aggregates for anhir-tissue77.

  MVR_DATASET=anhir python eval/anhir_core_table.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

os.environ["MVR_DATASET"] = "anhir"

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from setup import datasets
from setup.anhir.ingest import _open_zip, _zip_prefix
from setup.anhir.tissue_sample import tissue_of
from setup.coarse_to_fine import eval_runs, tre_eval
from setup.coarse_to_fine.eval_runs import read_cell_meta, read_runtime_s
from regWSI import paths as rpaths

BATCH_ID = "anhir-tissue77"
CANVAS_W = rpaths.CANVAS_W
CANVAS_H = rpaths.CANVAS_H
LAM_FFT = "fft"
LAM_SP = "superpoint_glue"
EST = "wendland"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_pair_rows() -> dict[int, dict]:
    raw = json.loads(datasets.ANHIR_PAIRS.read_text())
    rows = raw["pairs"] if isinstance(raw, dict) else raw
    return {int(p["id"]): p for p in rows}


def _he_src_wh(pair: dict, zf, prefix: str) -> tuple[int, int] | None:
    member = str(pair["target_image"])
    if prefix and not member.startswith(prefix):
        member = f"{prefix}{member}"
    try:
        with zf.open(member) as src:
            with Image.open(src) as im:
                w, h = im.size
        return int(w), int(h)
    except Exception:
        return None


def letterbox(src_w: int, src_h: int) -> dict:
    max_side = max(CANVAS_W, CANVAS_H)
    w0, h0 = float(src_w), float(src_h)
    w, h = w0, h0
    downsample = 1.0
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        w = max(1, int(round(w * s)))
        h = max(1, int(round(h * s)))
        downsample = max(w0 / float(w), h0 / float(h))
    fit = min(CANVAS_W / float(w), CANVAS_H / float(h))
    diag = math.hypot(w0, h0)
    return {
        "src_w": int(src_w),
        "src_h": int(src_h),
        "fit_scale": fit,
        "downsample": downsample,
        "diag_medium": diag,
        "rtre_factor": downsample / (fit * diag),
    }


def _he_wh_from_meta(pair_id: int) -> tuple[int, int] | None:
    path = rpaths.meta_json(pair_id)
    meta = _read_json(path) if path.is_file() else None
    if not meta:
        return None
    he = meta.get("he") or {}
    w = he.get("level0_w") or he.get("src_w")
    h = he.get("level0_h") or he.get("src_h")
    if w is None or h is None:
        return None
    return int(w), int(h)


def pair_geom(pair_ids: list[int]) -> dict[int, dict]:
    by_id = _load_pair_rows()
    out: dict[int, dict] = {}
    zf = None
    prefix = ""
    try:
        if datasets.ANHIR_ZIP.is_file():
            zf = _open_zip()
            prefix = _zip_prefix(zf)
        for pid in pair_ids:
            pair = by_id[pid]
            src_w = src_h = None
            src_w = src_h = None
            if zf is not None:
                src_w, src_h = _he_src_wh(pair, zf, prefix)
            if src_w is None:
                meta_wh = _he_wh_from_meta(pid)
                if meta_wh is not None:
                    src_w, src_h = meta_wh
            if src_w is None:
                src_w, src_h = int(pair["width"]), int(pair["height"])
            g = letterbox(int(src_w), int(src_h))
            g["case"] = pair.get("case")
            g["tissue"] = tissue_of(str(pair.get("case") or ""))
            g["scale"] = pair.get("scale")
            g["csv_diagonal"] = pair.get("diagonal")
            out[pid] = g
    finally:
        if zf is not None:
            zf.close()
    return out


def pair_stats(errs: list[float], factor: float, cap: float) -> dict | None:
    if not errs:
        return None
    a = np.asarray(errs, dtype=float)
    bad = ~np.isfinite(a) | (a > cap)
    a = np.where(bad, cap, a)
    r = a * factor
    return {
        "mean": float(np.mean(r)),
        "median": float(np.median(r)),
        "max": float(np.max(r)),
        "n": int(r.size),
        "n_capped": int(np.count_nonzero(bad)),
    }


def aggregate(per_pair: list[dict]) -> dict:
    means = [p["mean"] for p in per_pair]
    medians = [p["median"] for p in per_pair]
    maxes = [p["max"] for p in per_pair]
    def std(vals: list[float]) -> float:
        return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return {
        "n": len(per_pair),
        "AArTRE": float(np.mean(means)),
        "AArTRE_std": std(means),
        "AMrTRE": float(np.mean(medians)),
        "AMrTRE_std": std(medians),
        "MArTRE": float(np.median(means)),
        "MMrTRE": float(np.median(medians)),
        "AMxrTRE": float(np.mean(maxes)),
        "AMxrTRE_std": std(maxes),
        "MMxrTRE": float(np.median(maxes)),
        "n_capped": int(sum(p.get("n_capped") or 0 for p in per_pair)),
    }


def lam_uncached_runtime(batch_id: str, pair_id: int, lam: str) -> float | None:
    meta = read_cell_meta(batch_id, pair_id, lam, EST)
    if not meta:
        return None
    levels = meta.get("level_times") or {}
    l5 = levels.get("5") or levels.get(5) or {}
    if str(l5.get("cache") or "") != "compute":
        return None
    try:
        return float(meta["runtime_s"])
    except (KeyError, TypeError, ValueError):
        return None


def fmt_rtre(v: float) -> str:
    return f"{v:.4f}"


def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def fmt_time(v: float | None, n: int, n_ok: int) -> str:
    if v is None or n_ok == 0:
        return "---"
    if n_ok < n:
        return f"{v:.1f}\\textsuperscript{{*}}"
    return f"{v:.1f}"


def latex_table(rows: list[dict], n: int) -> str:
    body = []
    for r in rows:
        a = r["agg"]
        t = fmt_time(r.get("time_s"), n, int(r.get("time_n") or 0))
        body.append(
            f"    {r['tex']} & {fmt_pm(a['AArTRE'], a['AArTRE_std'])} & {fmt_pm(a['AMrTRE'], a['AMrTRE_std'])} & "
            f"{fmt_rtre(a['MArTRE'])} & {fmt_rtre(a['MMrTRE'])} & "
            f"{fmt_pm(a['AMxrTRE'], a['AMxrTRE_std'])} & {fmt_rtre(a['MMxrTRE'])} & {t} \\\\"
        )
    lines = "\n".join(body)
    return (
        r"\begin{tabular}{@{}lccccccc@{}}" + "\n"
        r"    \toprule" + "\n"
        r"    & \multicolumn{2}{c}{Average rTRE} & \multicolumn{2}{c}{Median rTRE}"
        r" & \multicolumn{2}{c}{Max rTRE} & Mean Time \\" + "\n"
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" + "\n"
        r"    Method & AArTRE & AMrTRE & MArTRE & MMrTRE & AMxrTRE & MMxrTRE & [s] \\" + "\n"
        r"    \midrule" + "\n"
        f"{lines}\n"
        r"    \bottomrule" + "\n"
        r"\end{tabular}"
    )


def cache_regwsi_tre(batch_id: str, pair_id: int) -> dict:
    path = eval_runs.regwsi_dir(batch_id, pair_id) / "tre.json"
    existing = _read_json(path)
    if (
        existing
        and existing.get("df_sample") == "he"
        and isinstance(existing.get("per_point"), list)
        and existing["per_point"]
    ):
        return existing
    datasets.set_active_dataset("anhir")
    base = tre_eval.compute_pair_baseline(pair_id)
    tre = base.get("regwsi") or {}
    if tre.get("error"):
        raise RuntimeError(f"pair {pair_id} regWSI TRE: {tre['error']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": tre.get("mean"),
        "median": tre.get("median"),
        "max": tre.get("max"),
        "p95": tre.get("p95"),
        "per_point": tre.get("per_point") or [],
        "ihc_warped": tre.get("ihc_warped") or [],
        "df_sample": "he",
    }
    path.write_text(json.dumps(payload))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=BATCH_ID)
    ap.add_argument("--out", type=Path, default=REPO / "eval" / "out" / "anhir_tissue77_rtres.json")
    args = ap.parse_args()
    datasets.set_active_dataset("anhir")
    man = eval_runs.read_manifest(args.batch)
    if not man:
        raise SystemExit(f"missing batch {args.batch}")
    pairs = [int(p) for p in man["pairs"]]
    geom = pair_geom(pairs)
    methods = [
        {"id": "initial", "tex": "Initial", "label": "Initial"},
        {
            "id": "fft_wendland",
            "tex": r"FFT $\times$ Wendland ($\varepsilon=0.2$)",
            "label": "FFT × Wendland (ε=0.2)",
            "lam": LAM_FFT,
        },
        {
            "id": "sp_wendland",
            "tex": r"SP+LG $\times$ Wendland ($\varepsilon=0.1$)",
            "label": "SP+LG × Wendland (ε=0.1)",
            "lam": LAM_SP,
        },
        {"id": "regwsi", "tex": "DeeperHistReg / regWSI", "label": "DeeperHistReg / regWSI"},
    ]
    per: dict[str, list[dict]] = {m["id"]: [] for m in methods}
    times: dict[str, list[float]] = {m["id"]: [] for m in methods}
    pair_rows: dict[str, dict] = {}

    for pid in pairs:
        factor = geom[pid]["rtre_factor"]
        w, h, _ = tre_eval.canvas_scale(pid)
        cap = math.hypot(w, h)
        points = tre_eval.load_landmarks(pid)
        none = tre_eval.tre_none(points, w, h).tolist() if points else []
        fft = _read_json(eval_runs.tre_path(args.batch, pid, LAM_FFT, EST)) or {}
        sp = _read_json(eval_runs.tre_path(args.batch, pid, LAM_SP, EST)) or {}
        rw = cache_regwsi_tre(args.batch, pid)
        stats = {
            "initial": pair_stats(none, factor, cap),
            "fft_wendland": pair_stats(list(fft.get("per_point") or []), factor, cap),
            "sp_wendland": pair_stats(list(sp.get("per_point") or []), factor, cap),
            "regwsi": pair_stats(list(rw.get("per_point") or []), factor, cap),
        }
        for mid, st in stats.items():
            if st is None:
                raise SystemExit(f"missing TRE for {mid} pair {pid}")
            per[mid].append(st)
        rec = {k: stats[k] for k in stats}
        rec["geom"] = geom[pid]
        rec["n_landmarks"] = len(points)
        pair_rows[str(pid)] = rec
        ft = lam_uncached_runtime(args.batch, pid, LAM_FFT)
        st = lam_uncached_runtime(args.batch, pid, LAM_SP)
        rwt = read_runtime_s(eval_runs.regwsi_runtime_path(args.batch, pid))
        if ft is not None:
            times["fft_wendland"].append(ft)
        if st is not None:
            times["sp_wendland"].append(st)
        if rwt is not None:
            times["regwsi"].append(rwt)

    out_rows = []
    for m in methods:
        agg = aggregate(per[m["id"]])
        tvals = times[m["id"]]
        row = {
            **m,
            "agg": agg,
            "time_s": float(np.mean(tvals)) if tvals else None,
            "time_n": len(tvals),
        }
        out_rows.append(row)
    out_rows.sort(key=lambda r: r["agg"]["AArTRE"])

    by_id = _load_pair_rows()
    sample = [
        {"id": pid, "case": by_id[pid].get("case"), "tissue": geom[pid]["tissue"]}
        for pid in pairs
    ]
    payload = {
        "batch_id": args.batch,
        "n_pairs": len(pairs),
        "rTRE": "TRE_canvas * downsample / (fit_scale * HE_medium_diagonal)",
        "tre_cap": "non-finite or > canvas-diagonal TRE clipped to hypot(canvas_w, canvas_h)",
        "canvas": [CANVAS_W, CANVAS_H],
        "hardware": "VPS 2×RTX 4090, 12 CPU workers, OMP=4; mean time uses uncached wall clocks only",
        "pairs": sample,
        "methods": [
            {
                "id": r["id"],
                "label": r["label"],
                **r["agg"],
                "time_s": r["time_s"],
                "time_n": r["time_n"],
            }
            for r in out_rows
        ],
        "per_pair": pair_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    tex = latex_table(out_rows, len(pairs))
    (args.out.with_suffix(".tex")).write_text(tex + "\n")
    print(json.dumps({"out": str(args.out), "methods": payload["methods"]}, indent=2))
    print(tex)


if __name__ == "__main__":
    main()
