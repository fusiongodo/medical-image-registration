"""
LaTeX table body for the random-angle SuperPoint fine-tune, stock vs tuned.

  python eval/table_sp_rot_final.py

Reads eval/out/sp_rot_final_within_test.json and eval/out/sp_rot_final_cross.json,
prints a summary to stdout and writes the tabular rows to
eval/out/sp_rot_rotinv_table.tex.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "eval" / "out"
CROSS_RESIZE = "1024"


def _cell_stats(summary: dict) -> dict:
    cells = summary["cells"]
    matches = [int(c.get("n_matches") or 0) for c in cells]
    inliers = [int(c.get("n_inliers") or 0) for c in cells]
    rot = [float(c["rot_err_deg"]) for c in cells if c.get("rot_err_deg") is not None]
    return {
        "matches": float(np.mean(matches)),
        "inliers": float(np.mean(inliers)),
        "rot_median": float(np.median(rot)) if rot else float("nan"),
        "n_pass": int(summary["n_pass"]),
        "n_total": int(summary["n_total"]),
        "recovery": 100.0 * float(summary["pass_rate"]),
    }


def main() -> None:
    within = json.loads((OUT / "sp_rot_final_within_test.json").read_text())
    cross = json.loads((OUT / "sp_rot_final_cross.json").read_text())

    rows = {}
    for label in ("stock", "tuned"):
        for kind in ("nn", "lg"):
            rows[("within", label, kind)] = _cell_stats(within["results"][label][kind])
            rows[("cross", label, kind)] = _cell_stats(
                cross["results"][label][CROSS_RESIZE][kind]
            )

    print(
        f"{'task':<7}{'weights':<8}{'matcher':<9}"
        f"{'matches':>9}{'inliers':>9}{'rot_med':>9}{'recovery':>12}"
    )
    for key, s in rows.items():
        task, label, kind = key
        print(
            f"{task:<7}{label:<8}{kind:<9}{s['matches']:>9.0f}{s['inliers']:>9.0f}"
            f"{s['rot_median']:>9.2f}{s['n_pass']:>6}/{s['n_total']:<3}"
            f"{s['recovery']:>5.0f}%"
        )

    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{two-way NN} & \\multicolumn{2}{c}{LightGlue} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "& & matches & recovery & matches & recovery \\\\",
        "\\midrule",
    ]
    for task, task_tex in (("within", "within-stain"), ("cross", "cross-stain")):
        for i, label in enumerate(("stock", "tuned")):
            nn, lg = rows[(task, label, "nn")], rows[(task, label, "lg")]
            head = task_tex if i == 0 else ""
            lines.append(
                f"{head} & {label} & {nn['matches']:.0f} & {nn['recovery']:.0f}\\% "
                f"& {lg['matches']:.0f} & {lg['recovery']:.0f}\\% \\\\"
            )
        if task == "within":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]

    n_within = rows[("within", "tuned", "nn")]["n_total"]
    n_cross = rows[("cross", "tuned", "nn")]["n_total"]
    body = "\n".join(lines) + "\n"
    dest = OUT / "sp_rot_rotinv_table.tex"
    dest.write_text(body)
    print(f"\nwithin trials={n_within}  cross trials={n_cross}  cross resize={CROSS_RESIZE}")
    print("wrote", dest)

    print("\ncross-stain resize sweep (recovery %):")
    for label in ("stock", "tuned"):
        for rz, per in cross["results"][label].items():
            print(
                f"  {label:<6} rz={rz:<5} "
                f"nn {100*per['nn']['pass_rate']:5.1f}%  lg {100*per['lg']['pass_rate']:5.1f}%"
            )


if __name__ == "__main__":
    main()
