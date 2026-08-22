"""CPU-only wall clock for one pair: regWSI vs FFT×Wendland vs SuperPoint×Wendland."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _pin_threads(n: int) -> None:
    s = str(max(1, int(n)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = s
    os.environ["MKL_NUM_THREADS"] = s
    os.environ["OPENBLAS_NUM_THREADS"] = s
    os.environ["NUMEXPR_NUM_THREADS"] = s
    os.environ["TORCH_NUM_THREADS"] = s


def _emit(stage: str, **kv) -> None:
    parts = [f"stage={stage}"]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def _load_payload(path: Path, pair_id: int, host: str, threads: int, eps: float) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "pair_id": int(pair_id),
        "host": host,
        "device": "cpu",
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "threads": int(threads),
        "wendland_eps": float(eps),
        "methods": {},
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id", type=int)
    ap.add_argument("--host", required=True, choices=("vps", "m4"))
    ap.add_argument(
        "--tag",
        default=None,
        help="write {host}_{tag}.json instead of {host}.json",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--wendland-eps", type=float, default=0.1)
    ap.add_argument(
        "--methods",
        default="regwsi,fft,superpoint_glue",
        help="comma-separated: regwsi,fft,superpoint_glue",
    )
    ap.add_argument("--dataset", default="anhir")
    args = ap.parse_args()
    _pin_threads(args.threads)

    from setup import datasets
    from setup.coarse_to_fine import eval_runs
    from setup.coarse_to_fine.eval_batch_cli import _fit_pair_lam_estimator
    from setup.coarse_to_fine.reg_branches import DEFAULT_BSPLINE_GRID, DEFAULT_BSPLINE_REG

    ds = datasets.normalize_dataset(args.dataset)
    datasets.set_active_dataset(ds)
    pair_id = int(args.pair_id)
    eps = float(args.wendland_eps)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tag = (args.tag or "").strip() or None
    path = eval_runs.cpu_runtime_path(pair_id, args.host, tag)
    if path.is_file():
        raise SystemExit(f"refusing to overwrite {path}")
    host_label = f"{args.host}_{tag}" if tag else args.host
    payload = _load_payload(path, pair_id, host_label, args.threads, eps)
    payload["threads"] = int(args.threads)
    payload["wendland_eps"] = eps
    payload["dataset"] = ds
    levels = list(eval_runs.DEFAULT_LEVELS)

    for method in methods:
        _emit("method", method=method, pair=pair_id, host=host_label)
        t0 = time.perf_counter()
        entry: dict = {"lam": None, "field_estimator": "wendland"}
        if method == "regwsi":
            from regWSI.register import register_pair

            register_pair(pair_id, persist_rigid=True)
            entry = {"runtime_s": time.perf_counter() - t0}
        elif method in ("fft", "superpoint_glue"):
            field, meta = _fit_pair_lam_estimator(
                pair_id,
                levels,
                method,
                "wendland",
                eps,
                DEFAULT_BSPLINE_GRID,
                DEFAULT_BSPLINE_REG,
                force=True,
            )
            del field
            entry = {
                "runtime_s": time.perf_counter() - t0,
                "lam": method,
                "field_estimator": "wendland",
                "lam_s": meta.get("lam_s"),
                "fit_s": meta.get("fit_s"),
                "init_fit_s": meta.get("init_fit_s"),
                "level_times": meta.get("level_times"),
            }
        else:
            raise SystemExit(f"unknown method {method!r}")
        payload["methods"][method] = entry
        _write(path, payload)
        _emit(
            "method_done",
            method=method,
            runtime_s=f"{entry['runtime_s']:.3f}",
            path=str(path.relative_to(REPO_ROOT)),
        )

    print(json.dumps(payload, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
