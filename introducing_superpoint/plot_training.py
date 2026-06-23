"""
Visualise a training run from its training_log.json.

Usage:
    python plot_training.py <run_dir_or_log_json> [--out fig.png]

If <path> is a directory it looks for training_log.json inside it.
Without --out the figure is shown interactively.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_log(path: Path) -> dict:
    if path.is_dir():
        path = path / "training_log.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _field(logs, key, default=None):
    return [e.get(key, default) for e in logs]


def _plot(log: dict, out_path: Path | None):
    logs   = log["epoch_logs"]
    epochs = _field(logs, "epoch")
    name   = log.get("name", "run")

    has_val = any(e.get("val_precision") is not None for e in logs)

    depth_keys = sorted({
        k for e in logs
        if e.get("val_kpis_by_depth")
        for k in e["val_kpis_by_depth"]
    })

    n_depth_rows = len(depth_keys)
    n_rows = 3 + (1 if n_depth_rows > 0 else 0)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    fig.suptitle(name, fontsize=14)

    def ax(row, col):
        return axes[row][col] if n_rows > 1 else axes[col]

    # --- row 0: loss curves ---
    loss_keys = [
        ("loss_total",      "total"),
        ("loss_descriptor", "descriptor"),
        ("loss_keypoint",   "keypoint"),
        ("loss_loc",        "loc"),
        ("loss_fn",         "fn"),
        ("loss_fp",         "fp"),
    ]
    panels = [
        (0, 0, ["loss_total", "loss_keypoint", "loss_descriptor"], "Losses"),
        (0, 1, ["loss_loc", "loss_fn", "loss_fp"],                 "KP sub-losses"),
    ]
    for row, col, keys, title in panels:
        a = ax(row, col)
        for k in keys:
            vals = _field(logs, k)
            label = k.replace("loss_", "")
            a.plot(epochs, vals, label=label)
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend(fontsize=7)

    ax(0, 2).axis("off")

    # --- row 1: train KPIs ---
    kpi_keys = [("precision", "repeatability"), ("recall", "repeatability"), ("repeatability", None)]
    kpi_panel = [
        (1, 0, "precision"),
        (1, 1, "recall"),
        (1, 2, "repeatability"),
    ]
    for row, col, key in kpi_panel:
        a = ax(row, col)
        a.plot(epochs, _field(logs, key), label=f"train {key}", color="steelblue")
        if has_val:
            val_key = f"val_{key}"
            val_vals = _field(logs, val_key)
            if any(v is not None for v in val_vals):
                val_ep  = [e for e, v in zip(epochs, val_vals) if v is not None]
                val_pts = [v for v in val_vals if v is not None]
                a.plot(val_ep, val_pts, label=f"val {key}", linestyle="--", color="tomato")
        a.set_title(key)
        a.set_xlabel("epoch")
        a.legend(fontsize=7)

    # --- row 2: val KPIs by depth (precision + recall + repeatability) ---
    if depth_keys:
        depth_panels = [
            (2, 0, "precision"),
            (2, 1, "recall"),
            (2, 2, "repeatability"),
        ]
        for row, col, metric in depth_panels:
            a = ax(row, col)
            for dk in depth_keys:
                vals = [
                    e["val_kpis_by_depth"][dk][metric]
                    if (e.get("val_kpis_by_depth") or {}).get(dk)
                    else None
                    for e in logs
                ]
                ep_  = [e for e, v in zip(epochs, vals) if v is not None]
                pts_ = [v for v in vals if v is not None]
                if ep_:
                    a.plot(ep_, pts_, label=f"d{dk}")
            a.set_title(f"val {metric} by depth")
            a.set_xlabel("epoch")
            a.legend(fontsize=7)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"saved → {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="run dir or training_log.json")
    parser.add_argument("--out", type=Path, default=None, help="output image path")
    args = parser.parse_args()

    log = _load_log(args.path)
    _plot(log, args.out)


if __name__ == "__main__":
    main()
