"""Tabulate matches / inliers / rigid-fit error per step, angle and matcher.

usage: python eval/peek_sp_rot_eval.py <..._eval.jsonl> [every]
Gate is rot_err <= 1.0 deg and trans_err_rel <= 0.055 (sp_rot_train_eval).
"""

import json
import sys


def main() -> None:
    path = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rows = [json.loads(line) for line in open(path) if line.strip()]
    print(f"{'step':>7} {'match':>7} {'ang':>5} {'n_match':>8} {'inlier':>7} {'rot_err':>8} {'trans':>7} {'pass':>5}")
    for r in rows:
        step = int(r.get("step") or 0)
        if every and step % every and step != rows[-1].get("step"):
            continue
        for kind in ("nn", "lg"):
            for c in (r.get(kind) or {}).get("cells") or []:
                rot = c.get("rot_err_deg")
                tr = c.get("trans_err_rel")
                print(
                    f"{step:>7} {kind:>7} {float(c['angle']):>5.0f} "
                    f"{str(c.get('n_matches')):>8} {str(c.get('n_inliers')):>7} "
                    f"{'-' if rot is None else format(float(rot), '.3f'):>8} "
                    f"{'-' if tr is None else format(float(tr), '.4f'):>7} "
                    f"{str(bool(c.get('auto_pass'))):>5}"
                )


if __name__ == "__main__":
    main()
