"""NMS × extract-resolution SP+LG rigid rotation sweep.

Scores each (pair, angle) with the GT gate:
  pass ⇔ |rot_err_deg| ≤ 1.0 AND trans_err_px / min(W,H) ≤ 0.055

Usage:
  .venv/bin/python3 setup/coarse_to_fine/sp_rot_config_sweep.py create \\
      --name nms-res --pairs 3,5,7,11,14
  .venv/bin/python3 setup/coarse_to_fine/sp_rot_config_sweep.py run <sweep_id>
  .venv/bin/python3 setup/coarse_to_fine/sp_rot_config_sweep.py summary <sweep_id>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import conf
from setup import datasets
from setup.coarse_to_fine import rigid_sp_lg
from setup.coarse_to_fine import sp_rot_bench as bench

SWEEP_ROOT = conf.PROJECT_ROOT / "data" / "sp_rot_sweeps"
DEFAULT_PAIRS = [3, 5, 7, 11, 14]
DEFAULT_NMS = [8, 12, 16, 20, 24]
DEFAULT_EXTRACT = [512, 768, 1024]
DEFAULT_PREVIEW_LEVEL = 2
MAX_ROT_ERR_DEG = 1.0
MAX_TRANS_ERR_REL = 0.055


def parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    seen: set[int] = set()
    uniq: list[int] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def emit(obj) -> None:
    print(bench.dumps(obj, indent=2))


def sweep_dir(sweep_id: str) -> Path:
    return SWEEP_ROOT / sweep_id


def manifest_path(sweep_id: str) -> Path:
    return sweep_dir(sweep_id) / "manifest.json"


def status_path(sweep_id: str) -> Path:
    return sweep_dir(sweep_id) / "status.json"


def summary_path(sweep_id: str) -> Path:
    return sweep_dir(sweep_id) / "summary.json"


def config_key(nms: int, extract: int) -> str:
    return f"nms{int(nms)}_res{int(extract)}"


def cell_dir(sweep_id: str, nms: int, extract: int, pair_id: int, angle: int) -> Path:
    return sweep_dir(sweep_id) / config_key(nms, extract) / str(int(pair_id)) / str(int(angle))


def gt_path(sweep_id: str, pair_id: int) -> Path:
    return sweep_dir(sweep_id) / "gt" / f"{int(pair_id)}.json"


def load_manifest(sweep_id: str) -> dict | None:
    p = manifest_path(sweep_id)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def write_status(sweep_id: str, st: dict) -> None:
    status_path(sweep_id).write_text(bench.dumps(st, indent=2))


def auto_pass(rot_err_deg: float | None, trans_err_rel: float | None, man: dict) -> bool | None:
    if rot_err_deg is None or trans_err_rel is None:
        return None
    max_rot = float(man.get("max_rot_err_deg", MAX_ROT_ERR_DEG))
    max_tr = float(man.get("max_trans_err_rel", MAX_TRANS_ERR_REL))
    return bool(rot_err_deg <= max_rot and trans_err_rel <= max_tr)


def cmd_create(args: argparse.Namespace) -> None:
    pairs = parse_int_list(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    nms_list = parse_int_list(args.nms) if args.nms else list(DEFAULT_NMS)
    extract_list = parse_int_list(args.extract) if args.extract else list(DEFAULT_EXTRACT)
    angles = parse_int_list(args.angles) if args.angles else list(bench.DEFAULT_ANGLES)
    if not pairs or not nms_list or not extract_list or not angles:
        emit({"ok": False, "error": "empty pairs/nms/extract/angles"})
        sys.exit(1)
    sid = bench.slugify(args.id or args.name)
    root = sweep_dir(sid)
    if root.exists():
        emit({"ok": False, "error": f"sweep already exists: {sid}"})
        sys.exit(1)
    root.mkdir(parents=True, exist_ok=True)
    total = len(pairs) * len(angles) * len(nms_list) * len(extract_list)
    man = {
        "id": sid,
        "name": (args.name or sid).strip(),
        "dataset": datasets.normalize_dataset(args.dataset),
        "pairs": pairs,
        "angles": angles,
        "nms_list": nms_list,
        "extract_list": extract_list,
        "preview_level": int(args.preview_level),
        "max_rot_err_deg": float(args.max_rot_err_deg),
        "max_trans_err_rel": float(args.max_trans_err_rel),
        "created_at": int(time.time()),
        "total_cells": total,
    }
    manifest_path(sid).write_text(bench.dumps(man, indent=2))
    write_status(
        sid,
        {
            "state": "created",
            "done": 0,
            "total": total,
            "failed": 0,
            "skipped": 0,
            "detail": None,
            "updated_at": int(time.time()),
        },
    )
    emit({"ok": True, "sweep": man})


def _prepare_gt(sweep_id: str, pair_id: int, dataset: str) -> dict:
    dest = gt_path(sweep_id, pair_id)
    if dest.is_file():
        return json.loads(dest.read_text())
    store = bench.ensure_gt_rigid(pair_id, dataset)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(bench.dumps(store, indent=2))
    return store


def _run_cell(
    sweep_id: str,
    man: dict,
    pair_id: int,
    angle: int,
    nms: int,
    extract: int,
    gt: dict,
    *,
    force: bool,
) -> dict:
    cell = cell_dir(sweep_id, nms, extract, pair_id, angle)
    result_path = cell / "result.json"
    if result_path.is_file() and not force:
        existing = json.loads(result_path.read_text())
        if not existing.get("error"):
            return {**existing, "_skipped": True}

    preview_level = int(man.get("preview_level") or DEFAULT_PREVIEW_LEVEL)
    hp = {**rigid_sp_lg.DEFAULT_HYPERPARAMS, "sp_nms_dist": int(nms)}
    t0 = time.time()
    try:
        result = rigid_sp_lg.run(
            pair_id,
            preview_level=preview_level,
            pre_rotation_deg=float(angle),
            hyperparams=hp,
            extract_resize=int(extract),
            write_artifacts=False,
        )
        stats = result.get("stats") or {}
        w = float(stats.get("width") or (gt.get("stats") or {}).get("width") or 1.0)
        h = float(stats.get("height") or (gt.get("stats") or {}).get("height") or 1.0)
        cmp_m = bench.compare_rigid_to_gt(
            result.get("rigid"),
            gt.get("rigid"),
            width=w,
            height=h,
        )
        tr_px = cmp_m.get("trans_err_px")
        min_wh = min(w, h) if min(w, h) > 0 else 1.0
        trans_rel = float(tr_px) / min_wh if tr_px is not None else None
        rot = cmp_m.get("rot_err_deg")
        passed = auto_pass(
            float(rot) if rot is not None else None,
            trans_rel,
            man,
        )
        out = {
            "pair_id": int(pair_id),
            "angle": int(angle),
            "sp_nms_dist": int(nms),
            "extract_resize": int(extract),
            "preview_level": preview_level,
            "n_matches": result.get("n_matches"),
            "n_inliers": result.get("n_inliers"),
            "hyperparams": result.get("hyperparams"),
            "rigid": result.get("rigid"),
            "stats": stats,
            **cmp_m,
            "trans_err_rel": trans_rel,
            "auto_pass": passed,
            "runtime_s": float(time.time() - t0),
            "ran_at": int(time.time()),
            "error": None,
        }
        cell.mkdir(parents=True, exist_ok=True)
        result_path.write_text(bench.dumps(out, indent=2))
        return out
    except Exception as e:
        out = {
            "pair_id": int(pair_id),
            "angle": int(angle),
            "sp_nms_dist": int(nms),
            "extract_resize": int(extract),
            "auto_pass": False,
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
            "runtime_s": float(time.time() - t0),
            "ran_at": int(time.time()),
        }
        cell.mkdir(parents=True, exist_ok=True)
        result_path.write_text(bench.dumps(out, indent=2))
        return out
    finally:
        rd = rigid_sp_lg.run_dir(pair_id)
        if rd.exists():
            shutil.rmtree(rd, ignore_errors=True)


def build_summary(sweep_id: str) -> dict:
    man = load_manifest(sweep_id)
    if not man:
        raise FileNotFoundError(sweep_id)
    pairs = [int(p) for p in man["pairs"]]
    angles = [int(a) for a in man["angles"]]
    nms_list = [int(n) for n in man["nms_list"]]
    extract_list = [int(r) for r in man["extract_list"]]
    table: dict[str, dict[str, dict]] = {}
    by_config_angle: dict[str, dict[str, dict]] = {}
    best = None
    for nms in nms_list:
        table[str(nms)] = {}
        for extract in extract_list:
            n_pass = 0
            n_total = 0
            n_error = 0
            rot_pass: list[float] = []
            tr_pass: list[float] = []
            angle_stats: dict[str, dict] = {}
            for ang in angles:
                angle_stats[str(ang)] = {"n": 0, "n_pass": 0, "n_error": 0}
            for pid in pairs:
                for ang in angles:
                    p = cell_dir(sweep_id, nms, extract, pid, ang) / "result.json"
                    if not p.is_file():
                        continue
                    res = json.loads(p.read_text())
                    n_total += 1
                    akey = str(ang)
                    angle_stats[akey]["n"] += 1
                    if res.get("error"):
                        n_error += 1
                        angle_stats[akey]["n_error"] += 1
                        continue
                    if res.get("auto_pass"):
                        n_pass += 1
                        angle_stats[akey]["n_pass"] += 1
                        if res.get("rot_err_deg") is not None:
                            rot_pass.append(float(res["rot_err_deg"]))
                        if res.get("trans_err_rel") is not None:
                            tr_pass.append(float(res["trans_err_rel"]))
            for akey, st in angle_stats.items():
                n = st["n"]
                st["pass_rate"] = (st["n_pass"] / n) if n else None
            cell = {
                "n_pass": n_pass,
                "n_total": n_total,
                "n_error": n_error,
                "pass_rate": (n_pass / n_total) if n_total else None,
                "mean_rot_err_pass": (sum(rot_pass) / len(rot_pass)) if rot_pass else None,
                "mean_trans_err_rel_pass": (sum(tr_pass) / len(tr_pass)) if tr_pass else None,
            }
            table[str(nms)][str(extract)] = cell
            ck = config_key(nms, extract)
            by_config_angle[ck] = angle_stats
            if cell["pass_rate"] is not None:
                cand = {
                    "sp_nms_dist": nms,
                    "extract_resize": extract,
                    **cell,
                }
                if best is None or (cand["pass_rate"] or 0) > (best["pass_rate"] or 0):
                    best = cand
    summary = {
        "sweep_id": sweep_id,
        "gate": {
            "max_rot_err_deg": man.get("max_rot_err_deg", MAX_ROT_ERR_DEG),
            "max_trans_err_rel": man.get("max_trans_err_rel", MAX_TRANS_ERR_REL),
        },
        "nms_list": nms_list,
        "extract_list": extract_list,
        "table": table,
        "by_config_angle": by_config_angle,
        "best": best,
        "updated_at": int(time.time()),
    }
    summary_path(sweep_id).write_text(bench.dumps(summary, indent=2))
    return summary


def print_table(summary: dict) -> None:
    nms_list = summary["nms_list"]
    extract_list = summary["extract_list"]
    table = summary["table"]
    hdr = "NMS\\res".ljust(8) + "".join(f"{r:>10}" for r in extract_list)
    print(hdr)
    print("-" * len(hdr))
    for nms in nms_list:
        row = f"{nms:<8}"
        for extract in extract_list:
            cell = (table.get(str(nms)) or {}).get(str(extract)) or {}
            pr = cell.get("pass_rate")
            if pr is None:
                row += f"{'—':>10}"
            else:
                row += f"{100.0 * pr:9.1f}%"
        print(row)
    best = summary.get("best")
    if best:
        print(
            f"\nbest: nms={best['sp_nms_dist']} extract={best['extract_resize']} "
            f"pass_rate={100.0 * best['pass_rate']:.1f}% "
            f"({best['n_pass']}/{best['n_total']}, errors={best['n_error']})"
        )
    gate = summary.get("gate") or {}
    print(
        f"gate: |rot|≤{gate.get('max_rot_err_deg')}°  "
        f"t_rel≤{gate.get('max_trans_err_rel')}"
    )


def cmd_run(args: argparse.Namespace) -> None:
    man = load_manifest(args.sweep_id)
    if not man:
        emit({"ok": False, "error": f"unknown sweep {args.sweep_id}"})
        sys.exit(1)
    ds = datasets.set_active_dataset(man.get("dataset") or "muromi")
    pairs = [int(p) for p in man["pairs"]]
    angles = [int(a) for a in man["angles"]]
    nms_list = [int(n) for n in man["nms_list"]]
    extract_list = [int(r) for r in man["extract_list"]]
    if args.pairs:
        want = set(parse_int_list(args.pairs))
        pairs = [p for p in pairs if p in want]
    if args.nms:
        want = set(parse_int_list(args.nms))
        nms_list = [n for n in nms_list if n in want]
    if args.extract:
        want = set(parse_int_list(args.extract))
        extract_list = [r for r in extract_list if r in want]
    force = bool(args.force)
    total = len(pairs) * len(angles) * len(nms_list) * len(extract_list)
    done = failed = skipped = 0
    write_status(
        args.sweep_id,
        {
            "state": "running",
            "done": 0,
            "total": total,
            "failed": 0,
            "skipped": 0,
            "detail": "start",
            "updated_at": int(time.time()),
        },
    )
    gts: dict[int, dict] = {}
    for pid in pairs:
        try:
            gts[pid] = _prepare_gt(args.sweep_id, pid, ds)
        except Exception as e:
            for nms in nms_list:
                for extract in extract_list:
                    for ang in angles:
                        cell = cell_dir(args.sweep_id, nms, extract, pid, ang)
                        cell.mkdir(parents=True, exist_ok=True)
                        (cell / "result.json").write_text(
                            bench.dumps(
                                {
                                    "pair_id": pid,
                                    "angle": ang,
                                    "sp_nms_dist": nms,
                                    "extract_resize": extract,
                                    "auto_pass": False,
                                    "error": f"gt failed: {e}",
                                    "ran_at": int(time.time()),
                                },
                                indent=2,
                            )
                        )
                        failed += 1
                        done += 1
            continue
        for nms in nms_list:
            for extract in extract_list:
                for ang in angles:
                    detail = f"nms={nms} res={extract} pair={pid} angle={ang}"
                    print(detail, flush=True)
                    out = _run_cell(
                        args.sweep_id,
                        man,
                        pid,
                        ang,
                        nms,
                        extract,
                        gts[pid],
                        force=force,
                    )
                    if out.get("_skipped"):
                        skipped += 1
                    elif out.get("error"):
                        failed += 1
                    done += 1
                    write_status(
                        args.sweep_id,
                        {
                            "state": "running",
                            "done": done,
                            "total": total,
                            "failed": failed,
                            "skipped": skipped,
                            "detail": detail,
                            "updated_at": int(time.time()),
                        },
                    )
    summary = build_summary(args.sweep_id)
    write_status(
        args.sweep_id,
        {
            "state": "done",
            "done": done,
            "total": total,
            "failed": failed,
            "skipped": skipped,
            "detail": "finished",
            "updated_at": int(time.time()),
        },
    )
    print_table(summary)
    emit(
        {
            "ok": True,
            "sweep_id": args.sweep_id,
            "done": done,
            "total": total,
            "failed": failed,
            "skipped": skipped,
            "best": summary.get("best"),
        }
    )


def cmd_summary(args: argparse.Namespace) -> None:
    summary = build_summary(args.sweep_id)
    print_table(summary)
    if args.json:
        emit(summary)


def cmd_list(_: argparse.Namespace) -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(SWEEP_ROOT.iterdir()):
        man = load_manifest(p.name)
        if man:
            runs.append(man)
    emit({"sweeps": runs})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--id", default=None)
    c.add_argument("--pairs", default=None, help="default 3,5,7,11,14")
    c.add_argument("--nms", default=None, help="default 8,12,16,20,24")
    c.add_argument("--extract", default=None, help="default 512,768,1024")
    c.add_argument("--angles", default=None, help="default 0..330/30")
    c.add_argument("--dataset", default="muromi")
    c.add_argument("--preview-level", type=int, default=DEFAULT_PREVIEW_LEVEL)
    c.add_argument("--max-rot-err-deg", type=float, default=MAX_ROT_ERR_DEG)
    c.add_argument("--max-trans-err-rel", type=float, default=MAX_TRANS_ERR_REL)

    r = sub.add_parser("run")
    r.add_argument("sweep_id")
    r.add_argument("--pairs", default=None)
    r.add_argument("--nms", default=None)
    r.add_argument("--extract", default=None)
    r.add_argument("--force", action="store_true")

    s = sub.add_parser("summary")
    s.add_argument("sweep_id")
    s.add_argument("--json", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "create":
        cmd_create(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "summary":
        cmd_summary(args)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
