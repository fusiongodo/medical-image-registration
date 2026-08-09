"""
SP+LightGlue rotation bench CLI.

Usage:
  python setup/coarse_to_fine/sp_rot_bench_cli.py list
  python setup/coarse_to_fine/sp_rot_bench_cli.py create --name smoke --pairs 0,1
  python setup/coarse_to_fine/sp_rot_bench_cli.py run <run_id>
  python setup/coarse_to_fine/sp_rot_bench_cli.py run <run_id> --pairs 0 --force
  python setup/coarse_to_fine/sp_rot_bench_cli.py status <run_id>
  python setup/coarse_to_fine/sp_rot_bench_cli.py summary <run_id>
  python setup/coarse_to_fine/sp_rot_bench_cli.py label <run_id> --pair 0 --angle 90 --label pass
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup import datasets
from setup.coarse_to_fine import sp_rot_bench as bench
from setup.coarse_to_fine import rigid_sp_lg


def parse_pairs_spec(spec: str) -> list[int]:
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


def emit(obj: dict) -> None:
    print(bench.dumps(obj), flush=True)


def progress(run_id: str, **kwargs) -> None:
    bits = [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    print(" ".join(bits), flush=True)
    st = bench.load_status(run_id) or {}
    for k, v in kwargs.items():
        if k in ("done", "total", "failed", "skipped") and v is not None:
            try:
                st[k] = int(v)
            except (TypeError, ValueError):
                st[k] = v
        elif k == "stage" and v is not None:
            st["state"] = str(v)
        elif k == "detail":
            st["detail"] = v
        elif k == "error":
            st["error"] = v
    st["updated_at"] = int(time.time())
    bench.write_status(run_id, st)


def cmd_list(_: argparse.Namespace) -> None:
    emit({"runs": bench.list_runs()})


def cmd_create(args: argparse.Namespace) -> None:
    pairs = parse_pairs_spec(args.pairs)
    try:
        man = bench.create_run(
            args.name,
            pairs,
            dataset=args.dataset,
            run_id=args.id,
            preview_level=args.preview_level,
            notes=args.notes,
        )
        emit({"ok": True, "run": man})
    except FileExistsError as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    try:
        emit(bench.matrix_status(args.run_id))
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def cmd_summary(args: argparse.Namespace) -> None:
    try:
        emit(bench.build_summary(args.run_id))
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def cmd_label(args: argparse.Namespace) -> None:
    try:
        entry = bench.save_label(
            args.run_id,
            args.pair,
            args.angle,
            args.label,
            note=args.note,
        )
        emit({"ok": True, "label": entry})
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def cmd_clear_labels(args: argparse.Namespace) -> None:
    try:
        store = bench.clear_labels(args.run_id)
        emit({"ok": True, "labels": store})
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def cmd_save_labels(args: argparse.Namespace) -> None:
    try:
        raw = args.labels_json
        if not raw and not sys.stdin.isatty():
            raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        labels = payload.get("labels", payload)
        store = bench.replace_labels(args.run_id, labels)
        try:
            summary = bench.build_summary(args.run_id)
        except Exception:
            summary = None
        emit({"ok": True, "labels": store, "summary": summary})
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def _prepare_gt(run_id: str, pair_id: int, dataset: str) -> dict:
    store = bench.ensure_gt_rigid(pair_id, dataset)
    dest = bench.gt_copy_path(run_id, pair_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(bench.dumps(store, indent=2))
    return store


def _run_cell(
    run_id: str,
    pair_id: int,
    angle: int,
    gt: dict,
    *,
    preview_level: int,
    force: bool,
) -> dict:
    cell = bench.cell_dir(run_id, pair_id, angle)
    result_path = cell / "result.json"
    if result_path.is_file() and not force:
        existing = bench.load_cell_result(run_id, pair_id, angle)
        if existing and not existing.get("error"):
            return {**existing, "_skipped": True}

    t0 = time.time()
    try:
        result = rigid_sp_lg.run(
            pair_id,
            preview_level=preview_level,
            pre_rotation_deg=float(angle),
        )
        src = rigid_sp_lg.run_dir(pair_id)
        bench.copy_run_artifacts(src, cell)

        stats = result.get("stats") or {}
        w = float(stats.get("width") or (gt.get("stats") or {}).get("width") or 1.0)
        h = float(stats.get("height") or (gt.get("stats") or {}).get("height") or 1.0)
        cmp_m = bench.compare_rigid_to_gt(
            result.get("rigid"),
            gt.get("rigid"),
            width=w,
            height=h,
        )
        out = {
            "pair_id": int(pair_id),
            "angle": int(angle),
            "pre_rotation_deg": float(angle),
            "n_matches": result.get("n_matches"),
            "n_inliers": result.get("n_inliers"),
            "rigid": result.get("rigid"),
            "rigid_prerot": result.get("rigid_prerot"),
            "stats": stats,
            "gt_source": gt.get("source") or gt.get("version"),
            "gt_df_path": gt.get("df_path"),
            **cmp_m,
            "runtime_s": float(time.time() - t0),
            "ran_at": int(time.time()),
            "error": None,
        }
        result_path.write_text(bench.dumps(out, indent=2))
        return out
    except Exception as e:
        out = {
            "pair_id": int(pair_id),
            "angle": int(angle),
            "pre_rotation_deg": float(angle),
            "error": str(e),
            "traceback": traceback.format_exc(limit=8),
            "runtime_s": float(time.time() - t0),
            "ran_at": int(time.time()),
        }
        cell.mkdir(parents=True, exist_ok=True)
        result_path.write_text(bench.dumps(out, indent=2))
        return out


def cmd_run(args: argparse.Namespace) -> None:
    man = bench.load_manifest(args.run_id)
    if not man:
        emit({"ok": False, "error": f"unknown run {args.run_id}"})
        sys.exit(1)

    ds = datasets.set_active_dataset(man.get("dataset") or "muromi")
    pairs = [int(p) for p in man.get("pairs") or []]
    if args.pairs:
        want = set(parse_pairs_spec(args.pairs))
        pairs = [p for p in pairs if p in want]
    angles = [int(a) for a in man.get("angles") or bench.DEFAULT_ANGLES]
    if args.angles:
        want_a = set(int(x) for x in args.angles.split(",") if x.strip() != "")
        angles = [a for a in angles if a in want_a]

    preview_level = int(args.preview_level if args.preview_level is not None else man.get("preview_level") or 2)
    force = bool(args.force)
    total = len(pairs) * len(angles)
    done = 0
    failed = 0
    skipped = 0

    progress(
        args.run_id,
        stage="running",
        done=0,
        total=total,
        failed=0,
        skipped=0,
        detail="start",
        error=None,
    )

    for pid in pairs:
        try:
            progress(
                args.run_id,
                stage="gt",
                done=done,
                total=total,
                pair=pid,
                detail=f"gt pair={pid}",
            )
            gt = _prepare_gt(args.run_id, pid, ds)
        except Exception as e:
            for ang in angles:
                cell = bench.cell_dir(args.run_id, pid, ang)
                cell.mkdir(parents=True, exist_ok=True)
                (cell / "result.json").write_text(
                    bench.dumps(
                        {
                            "pair_id": pid,
                            "angle": ang,
                            "error": f"gt failed: {e}",
                            "ran_at": int(time.time()),
                        },
                        indent=2,
                    )
                )
                failed += 1
                done += 1
                progress(
                    args.run_id,
                    stage="running",
                    done=done,
                    total=total,
                    failed=failed,
                    skipped=skipped,
                    pair=pid,
                    angle=ang,
                    detail=f"gt-fail pair={pid}",
                )
            continue

        for ang in angles:
            progress(
                args.run_id,
                stage="cell",
                done=done,
                total=total,
                failed=failed,
                skipped=skipped,
                pair=pid,
                angle=ang,
                detail=f"pair={pid} angle={ang}",
            )
            out = _run_cell(
                args.run_id,
                pid,
                ang,
                gt,
                preview_level=preview_level,
                force=force,
            )
            if out.get("_skipped"):
                skipped += 1
            elif out.get("error"):
                failed += 1
            done += 1
            progress(
                args.run_id,
                stage="running",
                done=done,
                total=total,
                failed=failed,
                skipped=skipped,
                pair=pid,
                angle=ang,
                detail=f"pair={pid} angle={ang}",
            )

    try:
        summary = bench.build_summary(args.run_id)
    except Exception:
        summary = None

    progress(
        args.run_id,
        stage="done",
        done=done,
        total=total,
        failed=failed,
        skipped=skipped,
        detail="finished",
        error=None,
    )
    emit(
        {
            "ok": True,
            "run_id": args.run_id,
            "done": done,
            "total": total,
            "failed": failed,
            "skipped": skipped,
            "summary": summary,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SP+LightGlue rotation bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list runs")

    c = sub.add_parser("create", help="create a run")
    c.add_argument("--name", required=True)
    c.add_argument("--pairs", required=True, help="e.g. 0,1,4 or 0-9")
    c.add_argument("--dataset", default="muromi")
    c.add_argument("--id", default=None)
    c.add_argument("--preview-level", type=int, default=2)
    c.add_argument("--notes", default=None)

    r = sub.add_parser("run", help="run SP+LG over pair×angle grid")
    r.add_argument("run_id")
    r.add_argument("--pairs", default=None)
    r.add_argument("--angles", default=None, help="comma angles subset")
    r.add_argument("--preview-level", type=int, default=None)
    r.add_argument("--force", action="store_true")

    s = sub.add_parser("status", help="matrix + status JSON")
    s.add_argument("run_id")

    u = sub.add_parser("summary", help="rebuild summary.json")
    u.add_argument("run_id")

    l = sub.add_parser("label", help="set human label for a cell")
    l.add_argument("run_id")
    l.add_argument("--pair", type=int, required=True)
    l.add_argument("--angle", type=int, required=True)
    l.add_argument("--label", required=True, choices=list(bench.LABELS))
    l.add_argument("--note", default=None)

    cl = sub.add_parser("clear-labels", help="wipe all human labels for a run")
    cl.add_argument("run_id")

    sl = sub.add_parser("save-labels", help="replace all labels from JSON")
    sl.add_argument("run_id")
    sl.add_argument(
        "--labels-json",
        default=None,
        help='JSON object {"labels":{"0:30":"pass",...}} or bare map',
    )

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "create":
        cmd_create(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "label":
        cmd_label(args)
    elif args.cmd == "clear-labels":
        cmd_clear_labels(args)
    elif args.cmd == "save-labels":
        cmd_save_labels(args)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
