"""
Plot the original-loss overfit run: detector CE split into FN/FP, desc loss,
and per-eval match counts / pass rate for two-way NN vs LightGlue.

  python eval/plot_sp_rot_overfit_original.py [run_stem]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "sp_rot_train" / "_overfit_original_loss"
FIG = ROOT / "figures"
MATCHERS = {"nn": ("C0", "two-way NN"), "lg": ("C3", "LightGlue")}


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


def _series(rows: list[dict], key: str):
    steps = [int(r["step"]) for r in rows if r.get(key) is not None]
    vals = [r[key] for r in rows if r.get(key) is not None]
    return steps, vals


def _plot(stem: str) -> None:
    loss = _rows(ROOT / f"{stem}.jsonl")
    ev = _rows(ROOT / f"{stem}_eval.jsonl")
    angles = sorted({float(c["angle"]) for r in ev for c in (r.get("nn") or {}).get("cells") or []})

    fig, (ax, ax_m, ax_k) = plt.subplots(
        3, 1, figsize=(7.6, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]}
    )

    for key, color, label in (
        ("loss_kp", "C2", "detector CE"),
        ("loss_fn", "C0", "CE on interest cells"),
        ("loss_fp", "C1", "CE on dustbin cells"),
        ("loss_desc", "C4", "desc hinge"),
    ):
        s, v = _series(loss, key)
        if s:
            ax.plot(s, v, color=color, lw=1.2, label=label)
    ax.set_ylabel("loss")
    ax.set_title(stem)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)

    styles = {a: ls for a, ls in zip(angles, ("-", "--", ":", "-."))}
    for kind, (color, label) in MATCHERS.items():
        for a in angles:
            steps, counts = [], []
            for r in ev:
                cell = next(
                    (
                        c
                        for c in (r.get(kind) or {}).get("cells") or []
                        if float(c["angle"]) == a
                    ),
                    None,
                )
                if cell is None:
                    continue
                steps.append(int(r["step"]))
                counts.append(cell.get("n_matches") or 0)
            if steps:
                ax_m.plot(
                    steps,
                    counts,
                    color=color,
                    ls=styles.get(a, "-"),
                    lw=1.2,
                    label=f"{label} {a:g} deg",
                )
    ax_m.set_ylabel("matches")
    ax_m.set_yscale("symlog", linthresh=10)
    ax_m.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax_m.grid(True, alpha=0.25)

    for kind, (color, label) in MATCHERS.items():
        steps = [int(r["step"]) for r in ev if (r.get(kind) or {}).get("n_total")]
        rate = [
            r[kind]["n_pass"] / r[kind]["n_total"]
            for r in ev
            if (r.get(kind) or {}).get("n_total")
        ]
        if steps:
            ax_k.plot(steps, rate, color=color, marker="o", ms=2.5, lw=1.0, label=label)
    ax_k.set_ylim(-0.05, 1.05)
    ax_k.set_ylabel("eval k/n")
    ax_k.set_xlabel("step")
    ax_k.legend(loc="upper right", fontsize=8, frameon=False)
    ax_k.grid(True, alpha=0.25)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG / f"loss_{stem}.{ext}"
        fig.savefig(path, dpi=160)
        print("wrote", path)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) > 1:
        stems = sys.argv[1:]
    else:
        stems = sorted(
            p.stem for p in ROOT.glob("overfit_product_*.jsonl") if "_eval" not in p.stem
        )
    if not stems:
        print(f"no run logs in {ROOT}", file=sys.stderr)
        sys.exit(1)
    for stem in stems:
        _plot(stem)


if __name__ == "__main__":
    main()
