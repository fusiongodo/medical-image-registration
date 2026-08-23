"""
Plot kp / desc / total loss and eval k/n for the 12_10 overfit trio.

  python eval/plot_sp_rot_overfit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "sp_rot_train" / "_overfit_nms"
FIG = ROOT / "figures"

RUNS = {
    "a12": {
        "loss": ROOT / "overfit_product_t1_a12_b12_d0_p0_12_10.jsonl",
        "eval": ROOT / "overfit_product_t1_a12_b12_d0_p0_12_10_eval.jsonl",
        "summary": ROOT / "overfit_product_t1_a12_b12_d0_p0_12_10_summary.json",
        "title": "a12  (0…330° / 30°)",
    },
    "a3": {
        "loss": ROOT / "overfit_product_t1_a3_b3_d0_p0_12_10_a3.jsonl",
        "eval": ROOT / "overfit_product_t1_a3_b3_d0_p0_12_10_a3_eval.jsonl",
        "summary": ROOT / "overfit_product_t1_a3_b3_d0_p0_12_10_a3_summary.json",
        "title": "a3  (0°, 120°, 240°)",
    },
    "a2": {
        "loss": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2.jsonl",
        "eval": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2_eval.jsonl",
        "summary": ROOT / "overfit_product_t1_a2_b2_d0_p0_12_10_a2_summary.json",
        "title": "a2  (0°, 120°)",
    },
}


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "start":
            continue
        out.append(row)
    return out


def _plot(name: str, spec: dict) -> None:
    loss = _rows(spec["loss"])
    ev = _rows(spec["eval"])
    man = json.loads(spec["summary"].read_text()) if spec["summary"].is_file() else {}
    collapse = man.get("first_collapse_step")

    steps = [int(r["step"]) for r in loss if r.get("step") is not None]
    kp_batch = [r.get("loss_kp") for r in loss]
    desc = [r.get("loss_desc") for r in loss]
    tot = [r.get("loss_total") for r in loss]
    mean_steps = [int(r["step"]) for r in loss if r.get("mean_kp") is not None]
    mean_kp = [r["mean_kp"] for r in loss if r.get("mean_kp") is not None]
    ev_s = [int(r["step"]) for r in ev]
    ev_k = [
        (r["n_pass"] / r["n_total"]) if r.get("n_total") else None
        for r in ev
    ]

    fig, (ax, ax_e) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(steps, kp_batch, color="0.75", lw=0.6, label="batch kp")
    if mean_steps:
        ax.plot(mean_steps, mean_kp, color="C0", lw=1.4, label="mean kp")
    ax.plot(steps, desc, color="C1", lw=1.2, label="desc")
    ax.plot(steps, tot, color="C2", lw=0.9, alpha=0.85, label="total")
    if collapse is not None:
        ax.axvline(float(collapse), color="0.35", ls="--", lw=0.9)
        ax.text(
            float(collapse),
            ax.get_ylim()[1] if False else 0.98,
            f" collapse {int(collapse)}",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="left",
            fontsize=8,
            color="0.35",
        )
    ax.set_ylabel("loss")
    ax.set_title(spec["title"])
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    if name == "a2":
        ax.set_ylim(0, 50)

    ax_e.plot(ev_s, ev_k, color="C3", marker="o", ms=2.5, lw=1.0)
    ax_e.set_ylim(-0.05, 1.05)
    ax_e.set_ylabel("eval k/n")
    ax_e.set_xlabel("step")
    ax_e.grid(True, alpha=0.25)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"loss_{name}.{ext}", dpi=160)
    plt.close(fig)


def main() -> None:
    missing = [k for k, s in RUNS.items() if not s["loss"].is_file()]
    if missing:
        print("missing loss logs:", missing, file=sys.stderr)
        sys.exit(1)
    for name, spec in RUNS.items():
        _plot(name, spec)
        print("wrote", FIG / f"loss_{name}.pdf")


if __name__ == "__main__":
    main()
