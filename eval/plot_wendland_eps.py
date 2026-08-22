"""
Wendland ε vs TRE for ANHIR pairs 0–19.

  python eval/plot_wendland_eps.py
  python eval/plot_wendland_eps.py --out eval/out/wendland_eps_tre.png
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import conf
from regWSI import paths as rpaths
from setup import datasets
from setup.anhir.ingest import _open_zip, _zip_prefix
from setup.anhir.tissue_sample import tissue_of

EVAL_RUNS = REPO / "data" / "eval_runs"
LAMS = ("fft", "superpoint_glue")
LAM_TITLE = {"fft": "FFT", "superpoint_glue": "SuperPoint + LightGlue"}
BATCH_IDS = (
    "anhir-full",
    "anhir-wen-e02",
    "anhir-wen-e03",
    "anhir-wen-e04",
    "anhir-wen-e05",
)

CANVAS_W = rpaths.CANVAS_W
CANVAS_H = rpaths.CANVAS_H
DIAG = math.hypot(CANVAS_W, CANVAS_H)
MPP_FULL_UM = {
    "COAD": 0.468,
    "lung-lesion": 0.174,
    "lung-lobes": 1.274,
    "mammary-gland": 2.294,
    "mice-kidney": 0.227,
    "gastric": 0.2528,
    "breast": 0.2528,
    "kidney": 0.2528,
}


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _tre_path(batch_id: str, pair_id: int, lam: str) -> Path:
    return EVAL_RUNS / batch_id / str(pair_id) / lam / "wendland" / "tre.json"


def _scale_frac(tag: str) -> float:
    m = re.match(r"scale-(\d+)pc$", tag or "")
    if not m:
        raise ValueError(f"unrecognized ANHIR scale tag {tag!r}")
    return int(m.group(1)) / 100.0


def _load_pairs() -> dict[int, dict]:
    raw = json.loads(datasets.ANHIR_PAIRS.read_text())
    rows = raw["pairs"] if isinstance(raw, dict) else raw
    return {int(p["id"]): p for p in rows}


def _he_src_wh(pair: dict, zf, prefix: str) -> tuple[int, int]:
    member = str(pair["target_image"])
    if prefix and not member.startswith(prefix):
        member = f"{prefix}{member}"
    try:
        with zf.open(member) as src:
            with Image.open(src) as im:
                w, h = im.size
        return int(w), int(h)
    except Exception:
        return int(pair["width"]), int(pair["height"])


def um_per_canvas_px(src_w: int, src_h: int, mpp_full: float, scale_frac: float) -> float:
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
    return (mpp_full / scale_frac) * downsample / fit


def pair_um_map(pair_ids: list[int]) -> dict[int, dict]:
    by_id = _load_pairs()
    out: dict[int, dict] = {}
    zf = None
    prefix = ""
    try:
        if datasets.ANHIR_ZIP.is_file():
            zf = _open_zip()
            prefix = _zip_prefix(zf)
        for pid in pair_ids:
            pair = by_id[pid]
            tissue = tissue_of(str(pair.get("case") or ""))
            mpp_full = MPP_FULL_UM[tissue]
            frac = _scale_frac(str(pair.get("scale") or ""))
            if zf is not None:
                src_w, src_h = _he_src_wh(pair, zf, prefix)
            else:
                src_w, src_h = int(pair["width"]), int(pair["height"])
            um = um_per_canvas_px(src_w, src_h, mpp_full, frac)
            out[pid] = {
                "case": pair.get("case"),
                "tissue": tissue,
                "scale": pair.get("scale"),
                "scale_frac": frac,
                "mpp_full_um": mpp_full,
                "src_w": src_w,
                "src_h": src_h,
                "um_per_canvas_px": um,
            }
    finally:
        if zf is not None:
            zf.close()
    return out


def load_sweep() -> dict:
    points: list[dict] = []
    pair_sets: list[set[int]] = []
    for batch_id in BATCH_IDS:
        man = _read_json(EVAL_RUNS / batch_id / "manifest.json")
        if not man:
            raise FileNotFoundError(f"missing {batch_id}/manifest.json")
        cfg = man.get("config") or {}
        eps = float(cfg["wendland_eps"])
        pairs = [int(p) for p in man.get("pairs") or []]
        pair_sets.append(set(pairs))
        points.append({"eps": eps, "batch_id": batch_id, "pairs": pairs})

    shared = set.intersection(*pair_sets) if pair_sets else set()
    complete: set[int] | None = None
    for p in points:
        ok = set()
        for pair_id in sorted(shared):
            if all(_tre_path(p["batch_id"], pair_id, lam).is_file() for lam in LAMS):
                ok.add(pair_id)
        complete = ok if complete is None else complete & ok
    pairs = sorted(complete or [])

    series: dict[str, list[dict]] = {lam: [] for lam in LAMS}
    for p in sorted(points, key=lambda x: x["eps"]):
        for lam in LAMS:
            means: list[float] = []
            medians: list[float] = []
            p95s: list[float] = []
            for pair_id in pairs:
                tre = _read_json(_tre_path(p["batch_id"], pair_id, lam)) or {}
                means.append(float(tre["mean"]))
                medians.append(float(tre["median"]))
                p95s.append(float(tre["p95"]))
            series[lam].append(
                {
                    "eps": p["eps"],
                    "batch_id": p["batch_id"],
                    "n": len(means),
                    "mean_of_means": _mean(means),
                    "mean_of_medians": _mean(medians),
                    "mean_of_p95": _mean(p95s),
                    "pair_means": means,
                    "pair_medians": medians,
                    "pair_p95s": p95s,
                }
            )
    return {
        "canvas": [CANVAS_W, CANVAS_H],
        "diagonal": DIAG,
        "level": rpaths.LEVEL,
        "tile": [conf.CNN_INPUT_WIDTH, conf.CNN_INPUT_HEIGHT],
        "pairs": pairs,
        "unit": "L5 canvas px",
        "series": series,
    }


def to_um(data: dict) -> dict:
    meta = pair_um_map(data["pairs"])
    scales = [float(meta[pid]["um_per_canvas_px"]) for pid in data["pairs"]]
    out = copy.deepcopy(data)
    out["unit"] = "µm"
    out["um_per_pair"] = {str(pid): meta[pid] for pid in data["pairs"]}
    for rows in out["series"].values():
        for row in rows:
            means = [v * s for v, s in zip(row["pair_means"], scales)]
            medians = [v * s for v, s in zip(row["pair_medians"], scales)]
            p95s = [v * s for v, s in zip(row["pair_p95s"], scales)]
            row["pair_means"] = means
            row["pair_medians"] = medians
            row["pair_p95s"] = p95s
            row["mean_of_means"] = _mean(means)
            row["mean_of_medians"] = _mean(medians)
            row["mean_of_p95"] = _mean(p95s)
    return out


def _draw_panel(ax, rows: list[dict], ylabel: str, *, legend: bool) -> None:
    xs = [r["eps"] for r in rows]
    med = [r["mean_of_medians"] for r in rows]
    avg = [r["mean_of_means"] for r in rows]
    ax.plot(xs, med, "o-", color="#1d4ed8", label="mean of medians")
    ax.plot(xs, avg, "s--", color="#b45309", label="mean of means")
    ax.set_xlabel("Wendland support $\\varepsilon$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
    ys = [v for v in med + avg if v is not None]
    lo, hi = min(ys), max(ys)
    pad = max(4.0, (hi - lo) * 0.18)
    ax.set_ylim(lo - pad, hi + pad)
    if legend:
        ax.legend(loc="best", frameon=False, fontsize=7)


def plot_sweep(data: dict, out: Path, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=False)
    for ax, lam in zip(axes, LAMS):
        ax.set_title(LAM_TITLE[lam])
        _draw_panel(ax, data["series"][lam], ylabel, legend=False)
    axes[1].legend(loc="best", frameon=False, fontsize=8)
    n = len(data["pairs"])
    fig.suptitle(
        f"ANHIR pairs {data['pairs'][0]}–{data['pairs'][-1]} (n={n})  ·  "
        f"L{data['level']}  {data['canvas'][0]}×{data['canvas'][1]}  "
        f"({2 ** data['level']}×{data['tile'][0]} × {2 ** data['level']}×{data['tile'][1]})"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_paper_panels(data: dict, out_dir: Path, ylabel: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"fft": "wendland_eps_fft_um.pdf", "superpoint_glue": "wendland_eps_sp_um.pdf"}
    written: dict[str, str] = {}
    for lam, name in names.items():
        fig, ax = plt.subplots(figsize=(3.35, 2.45))
        _draw_panel(ax, data["series"][lam], ylabel, legend=True)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        written[lam] = str(path)
    return written


def _write(data: dict, out: Path, ylabel: str) -> dict:
    plot_sweep(data, out, ylabel)
    table = {
        "unit": data["unit"],
        "canvas": data["canvas"],
        "diagonal": data["diagonal"],
        "level": data["level"],
        "tile": data["tile"],
        "pairs": data["pairs"],
        "series": {
            lam: [
                {
                    "eps": r["eps"],
                    "batch_id": r["batch_id"],
                    "n": r["n"],
                    "mean_of_medians": r["mean_of_medians"],
                    "mean_of_means": r["mean_of_means"],
                    "mean_of_p95": r["mean_of_p95"],
                }
                for r in rows
            ]
            for lam, rows in data["series"].items()
        },
    }
    if "um_per_pair" in data:
        table["um_per_pair"] = data["um_per_pair"]
    summary = out.with_suffix(".json")
    summary.write_text(json.dumps(table, indent=2))
    return {
        "png": str(out),
        "pdf": str(out.with_suffix(".pdf")),
        "json": str(summary),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "eval" / "out" / "wendland_eps_tre.png",
    )
    ap.add_argument(
        "--out-um",
        type=Path,
        default=REPO / "eval" / "out" / "wendland_eps_tre_um.png",
    )
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=REPO.parent / "paper" / "figures",
    )
    args = ap.parse_args()
    data = load_sweep()
    px = _write(data, args.out, "TRE (L5 canvas px)")
    data_um = to_um(data)
    um = _write(data_um, args.out_um, "TRE (µm)")
    paper = plot_paper_panels(data_um, args.paper_dir, r"TRE ($\mathrm{\mu}$m)")
    print(
        json.dumps(
            {
                "px": px,
                "um": um,
                "paper": paper,
                "series_um": {
                    lam: [
                        {
                            "eps": r["eps"],
                            "mean_of_medians": r["mean_of_medians"],
                            "mean_of_means": r["mean_of_means"],
                        }
                        for r in rows
                    ]
                    for lam, rows in data_um["series"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
