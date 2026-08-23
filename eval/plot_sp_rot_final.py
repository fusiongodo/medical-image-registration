"""
Training-curve figure for the random-angle SuperPoint fine-tune.

  python eval/plot_sp_rot_final.py

Top panel: the two halves of the detector cross-entropy (interest cells vs dustbin
cells) and the descriptor hinge, all on a log axis since they differ by ~2 decades.
Bottom panel: within-stain rigid-recovery rate on the 12 held-out validation crops
(144 crop/angle trials per point, LightGlue), against the stock-weight baseline.

Writes eval/out/sp_rot_rotinv_training.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "data" / "sp_rot_train" / "rot_rand_d4"
OUT = REPO / "eval" / "out"
STEM = "sp_rot_rotinv_training"

COLORS = {"fn": "#c0392b", "fp": "#e67e22", "desc": "#2c7fb8", "lg": "#2c7fb8"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _ema(ys: list[float], alpha: float = 0.12) -> list[float]:
    out, acc = [], None
    for y in ys:
        acc = y if acc is None else alpha * y + (1.0 - alpha) * acc
        out.append(acc)
    return out


def main() -> None:
    loss = _read_jsonl(RUN / "logs" / "loss.jsonl")
    evals = _read_jsonl(RUN / "logs" / "eval.jsonl")
    val = sorted(
        (r for r in evals if r.get("kind") == "val" and r.get("match_kind") == "lg"),
        key=lambda r: r["step"],
    )

    stock_lg = stock_nn = None
    wj = OUT / "sp_rot_final_within_val.json"
    if wj.is_file():
        res = json.loads(wj.read_text())["results"]["stock"]
        stock_lg = res["lg"]["pass_rate"]
        stock_nn = res["nn"]["pass_rate"]

    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.8})
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True)

    steps = [r["step"] for r in loss]
    series = [
        ("loss_fn", "CE, interest cells", COLORS["fn"]),
        ("loss_fp", "CE, dustbin cells", COLORS["fp"]),
        ("loss_desc", "descriptor hinge", COLORS["desc"]),
    ]
    for key, label, color in series:
        ys = [r[key] for r in loss]
        ax0.plot(steps, ys, color=color, alpha=0.18, lw=0.6)
        ax0.plot(steps, _ema(ys), color=color, lw=1.3, label=label)
    ax0.set_yscale("log")
    ax0.set_ylabel("loss (nats)")
    ax0.legend(loc="center left", frameon=False, ncol=1)
    ax0.grid(alpha=0.25, lw=0.4)

    vs = [r["step"] for r in val]
    vy = [100.0 * r["pass_rate"] for r in val]
    ax1.plot(vs, vy, color=COLORS["lg"], lw=1.3, marker="o", ms=2.0, label="LightGlue, tuned")
    if stock_lg is not None:
        ax1.axhline(
            100.0 * stock_lg,
            color=COLORS["lg"],
            ls="--",
            lw=1.0,
            label=f"LightGlue, stock ({100*stock_lg:.0f}%)",
        )
    if stock_nn is not None:
        ax1.axhline(
            100.0 * stock_nn,
            color="0.35",
            ls=":",
            lw=1.0,
            label=f"two-way NN, stock ({100*stock_nn:.0f}%)",
        )
    ax1.set_ylim(0, 103)
    ax1.set_xlabel("step")
    ax1.set_ylabel("rigid recovery (%)")
    ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.38), frameon=False)
    ax1.grid(alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = OUT / f"{STEM}.{ext}"
        fig.savefig(dest, dpi=220)
        print("wrote", dest)
    plt.close(fig)


if __name__ == "__main__":
    main()
