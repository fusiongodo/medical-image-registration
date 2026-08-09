"""
Eval batch CLI: create / list / run LAM × field-estimator grids.

Usage:
  python setup/coarse_to_fine/eval_batch_cli.py list
  python setup/coarse_to_fine/eval_batch_cli.py create --name demo --pairs 0,1,4,16
  python setup/coarse_to_fine/eval_batch_cli.py run <batch_id>
  python setup/coarse_to_fine/eval_batch_cli.py run <batch_id> --pairs 0-9 --skip-ingest --shard-id 0
  python setup/coarse_to_fine/eval_batch_cli.py run-parallel <batch_id> --workers 10 --skip-ingest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from setup import datasets
from setup.coarse_to_fine import annotations, deskew, eval_runs, masks, tre_eval
from setup.coarse_to_fine.field import (
    Candidate,
    Field,
    candidate_from_dict,
    fit_field,
    fit_gated,
    tau_for_keep,
)
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.reg_branches import cache_path, normalize_estimator, normalize_lam
from setup.coarse_to_fine.run import cache_candidates


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


def split_pair_shards(pairs: list[int], workers: int) -> list[list[int]]:
    if not pairs:
        return []
    n_workers = max(1, min(int(workers), len(pairs)))
    base, rem = divmod(len(pairs), n_workers)
    shards: list[list[int]] = []
    i = 0
    for w in range(n_workers):
        size = base + (1 if w < rem else 0)
        shards.append(list(pairs[i : i + size]))
        i += size
    return [s for s in shards if s]


def _emit(stage: str, **kv) -> None:
    parts = [f"stage={stage}"]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def _load_level_candidates(pair_id: int, level: int, lam: str) -> list[Candidate]:
    path = cache_path(pair_id, level, lam)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [candidate_from_dict(level, d) for d in payload.get("candidates", [])]


def _ensure_lam_caches(
    pair_id: int,
    levels: list[int],
    lam: str,
    force: bool,
) -> None:
    for level in sorted(levels):
        path = cache_path(pair_id, level, lam)
        if path.exists() and not force:
            _emit("cache_hit", pair=pair_id, lam=lam, level=level)
            continue
        _emit("cache_compute", pair=pair_id, lam=lam, level=level)
        cache_candidates(pair_id, level, levels, float("inf"), lam=lam)


def _fit_pair_lam_estimator(
    pair_id: int,
    levels: list[int],
    lam: str,
    estimator: str,
    wendland_eps: float,
    bspline_grid: int,
    bspline_reg: float,
) -> tuple[Field, dict]:
    import setup.coarse_to_fine.field as field_mod

    field_mod.WENDLAND_EPS = float(wendland_eps)
    field_mod.BSPLINE_GRID = int(bspline_grid)
    field_mod.BSPLINE_REG = float(bspline_reg)

    entries = annotations.load(pair_id)
    mask_entries = masks.load(pair_id)
    field = fit_field(
        annotations.to_anchors(entries, up_to_level=min(levels) - 1),
        field_estimator=estimator,
        wendland_epsilon=wendland_eps,
        bspline_grid=bspline_grid,
        bspline_reg=bspline_reg,
    )

    total_kept = 0
    total_seen = 0
    level_gates: dict[str, dict] = {}
    for level in sorted(levels):
        cands = _load_level_candidates(pair_id, level, lam)
        if not cands:
            continue
        masked = masks.masked_at(mask_entries, level, [c.tile_loc for c in cands])
        excluded = {
            e["tile_loc"]
            for e in entries
            if int(e["level"]) == level and e.get("type") == "exclude"
        }
        fit_cands = [
            c for c in cands if c.tile_loc not in masked and c.tile_loc not in excluded
        ]
        human = annotations.to_anchors(entries, up_to_level=level)
        keep = eval_runs.keep_for_level(level)
        if keep >= 1.0:
            tau = float("inf")
        else:
            tau = tau_for_keep(human, fit_cands, keep, field_estimator=estimator)
        field, kept = fit_gated(
            human, fit_cands, tau, field_estimator=estimator
        )
        level_gates[str(level)] = {
            "exclude_pct": 1.0 - keep,
            "keep": keep,
            "tau": None if tau == float("inf") else float(tau),
            "n_kept": len(kept),
            "n_seen": len(fit_cands),
        }
        total_kept += len(kept)
        total_seen += len(fit_cands)

    meta = {
        "lam": lam,
        "field_estimator": estimator,
        "exclude_pct_by_level": {
            str(k): float(v) for k, v in eval_runs.EXCLUDE_PCT_BY_LEVEL.items()
        },
        "level_gates": level_gates,
        "levels": levels,
        "n_kept": total_kept,
        "n_seen": total_seen,
        "n_human": len(entries),
        "wendland_eps": wendland_eps,
        "bspline_grid": bspline_grid,
        "bspline_reg": bspline_reg,
        "saved_depth": eval_runs.EVAL_DEPTH,
    }
    return field, meta


def _write_cell(
    batch_id: str,
    pair_id: int,
    lam: str,
    estimator: str,
    field: Field,
    meta: dict,
    runtime_s: float | None = None,
) -> dict:
    cell = eval_runs.cell_dir(batch_id, pair_id, lam, estimator)
    cell.mkdir(parents=True, exist_ok=True)
    ds = datasets.active_dataset()

    out_path = eval_runs.field_l5_path(batch_id, pair_id, lam, estimator)
    depths_out = {
        str(d): field.predict_tile_px(d) for d in range(eval_runs.EVAL_DEPTH + 1)
    }
    identity = (
        datasets.pair_fingerprint(pair_id, ds)
        if ds == "acrobat"
        else pair_fingerprint(pair_id)
    )
    payload = {
        "pair_id": pair_id,
        "dataset": ds,
        "identity": identity,
        "fit_depth": eval_runs.EVAL_DEPTH,
        "depths": depths_out,
        **meta,
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")))

    deskew_store = deskew.load(pair_id)
    if deskew_store:
        (cell / "deskew.json").write_text(json.dumps(deskew_store, separators=(",", ":")))

    rigid_path = datasets.rigid_path(pair_id, ds)
    if rigid_path.is_file():
        shutil_copy = rigid_path.read_text()
        (cell / "rigid.json").write_text(shutil_copy)

    if runtime_s is not None:
        meta = {**meta, "runtime_s": float(runtime_s)}

    eval_runs.meta_path(batch_id, pair_id, lam, estimator).write_text(
        json.dumps(meta, indent=2)
    )

    points = tre_eval.load_landmarks(pair_id)
    w, h, scale = tre_eval.canvas_scale(pair_id)
    if points:
        errs = tre_eval.tre_field_file(
            points,
            out_path,
            w,
            h,
            scale,
            rigid_path=rigid_path if rigid_path.is_file() else None,
        )
        tre = tre_eval.annotate_tile_means(tre_eval.stats(errs), scale)
    else:
        tre = tre_eval.empty_err("no landmarks")
    tre.update(
        {
            "lam": lam,
            "field_estimator": estimator,
            "n": len(points),
            "runtime_s": meta.get("runtime_s"),
            "config_fingerprint": meta.get("config_fingerprint"),
        }
    )
    eval_runs.tre_path(batch_id, pair_id, lam, estimator).write_text(
        json.dumps(tre, indent=2)
    )
    return tre


def run_batch(
    batch_id: str,
    *,
    pair_ids: list[int] | None = None,
    skip_ingest: bool = False,
    shard_id: int | None = None,
) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        raise FileNotFoundError(f"no batch {batch_id}")

    ds_name = datasets.normalize_dataset(manifest.get("dataset"))
    datasets.set_active_dataset(ds_name)

    manifest_pairs = [int(p) for p in manifest["pairs"]]
    allowed = set(manifest_pairs)
    if pair_ids is None:
        pairs = list(manifest_pairs)
    else:
        pairs = [int(p) for p in pair_ids if int(p) in allowed]
        missing = sorted({int(p) for p in pair_ids} - allowed)
        if missing:
            _emit(
                "pairs_ignored",
                n=len(missing),
                sample=",".join(str(p) for p in missing[:8]),
            )
    if not pairs:
        raise ValueError(f"no pairs to run for batch {batch_id}")

    lams = [normalize_lam(x) for x in manifest.get("lams") or []]
    estimators = [normalize_estimator(x) for x in manifest.get("estimators") or []]
    cfg = {**eval_runs.default_config(), **(manifest.get("config") or {})}
    cell_cfg = eval_runs.cell_config(cfg)
    fp = eval_runs.config_fingerprint(cell_cfg)
    levels = sorted(int(x) for x in cell_cfg["levels"])
    force = bool(cfg.get("force")) or bool(manifest.get("config", {}).get("force"))
    wendland_eps = float(cell_cfg["wendland_eps"])
    bspline_grid = int(cell_cfg["bspline_grid"])
    bspline_reg = float(cell_cfg["bspline_reg"])

    jobs = [(p, lam, est) for p in pairs for lam in lams for est in estimators]
    total = len(jobs) + (len(pairs) if ds_name == "acrobat" else 0)
    done = 0
    started_at = int(time.time())

    def push_status(
        *,
        state: str,
        detail: str,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        payload: dict = {
            "state": state,
            "done": done,
            "total": total,
            "detail": detail,
            "error": error,
            "pairs": pairs,
            "started_at": started_at,
        }
        if finished:
            payload["finished_at"] = int(time.time())
        if shard_id is not None:
            eval_runs.write_shard_status(batch_id, shard_id, payload)
        else:
            eval_runs.write_status(batch_id, payload)

    push_status(
        state="running",
        detail="ingest" if (ds_name == "acrobat" and not skip_ingest) else "start",
    )

    if ds_name == "acrobat" and not skip_ingest:
        from setup.acrobat.ingest import ingest

        _emit("ingest", dataset=ds_name, pairs=len(pairs), done=0, total=total)
        ingest_result = ingest(unzip=True, pair_ids=pairs, force=False)
        for err in ingest_result.get("errors") or []:
            _emit(
                "ingest_skip",
                pair=err.get("pair_id"),
                err=str(err.get("error", "")).replace(" ", "_")[:120],
            )
        _emit(
            "ingest_done",
            exported=ingest_result.get("exported"),
            failed=ingest_result.get("failed"),
        )

    _emit(
        "start",
        batch=batch_id,
        dataset=ds_name,
        total=total,
        pairs=len(pairs),
        shard=shard_id if shard_id is not None else "-",
        config_fp=fp,
    )

    try:
        if ds_name == "acrobat":
            from regWSI import paths as rpaths
            from regWSI.register import register_pair

            for pair_id in pairs:
                detail = f"pair={pair_id} stage=regwsi"
                _emit("regwsi", pair=pair_id, done=done, total=total)
                push_status(state="running", detail=detail)
                df = rpaths.displacement_field(pair_id)
                rigid = datasets.rigid_path(pair_id, "acrobat")
                he = rpaths.he_tiff(pair_id)
                ihc = rpaths.ihc_tiff(pair_id)
                if not he.is_file() or not ihc.is_file():
                    _emit("regwsi_skip", pair=pair_id, reason="missing_inputs")
                    done += 1
                    push_status(state="running", detail=detail)
                    continue
                if df.is_file() and rigid.is_file() and not force:
                    _emit("regwsi_skip", pair=pair_id)
                else:
                    try:
                        t0 = time.perf_counter()
                        register_pair(pair_id, persist_rigid=True)
                        runtime_s = time.perf_counter() - t0
                        rd = eval_runs.regwsi_dir(batch_id, pair_id)
                        rd.mkdir(parents=True, exist_ok=True)
                        (rd / "runtime.json").write_text(
                            json.dumps(
                                {"runtime_s": runtime_s, "pair_id": pair_id}, indent=2
                            )
                        )
                        _emit(
                            "regwsi_done",
                            pair=pair_id,
                            runtime_s=f"{runtime_s:.3f}",
                        )
                    except Exception as e:
                        _emit(
                            "regwsi_error",
                            pair=pair_id,
                            err=str(e).replace(" ", "_")[:160],
                        )
                done += 1
                push_status(state="running", detail=detail)

        for pair_id, lam, estimator in jobs:
            detail = f"pair={pair_id} lam={lam} estimator={estimator}"
            _emit(
                "cell",
                pair=pair_id,
                lam=lam,
                estimator=estimator,
                done=done,
                total=total,
            )
            push_status(state="running", detail=detail)

            if (
                eval_runs.cell_complete(batch_id, pair_id, lam, estimator, cell_cfg)
                and not force
            ):
                _emit("skip", pair=pair_id, lam=lam, estimator=estimator)
                done += 1
                push_status(state="running", detail=detail)
                continue

            t0 = time.perf_counter()
            _ensure_lam_caches(pair_id, levels, lam, force=force)
            field, meta = _fit_pair_lam_estimator(
                pair_id,
                levels,
                lam,
                estimator,
                wendland_eps,
                bspline_grid,
                bspline_reg,
            )
            meta = {
                **meta,
                "config_fingerprint": fp,
                "cell_config": cell_cfg,
            }
            runtime_s = time.perf_counter() - t0
            _write_cell(
                batch_id,
                pair_id,
                lam,
                estimator,
                field,
                meta,
                runtime_s=runtime_s,
            )
            _emit(
                "cell_done",
                pair=pair_id,
                lam=lam,
                estimator=estimator,
                runtime_s=f"{runtime_s:.3f}",
            )
            done += 1
            push_status(state="running", detail=detail)

        push_status(state="done", detail="complete", finished=True)
        _emit("done", batch=batch_id, total=total, shard=shard_id if shard_id is not None else "-")
        return {
            "ok": True,
            "batch_id": batch_id,
            "total": total,
            "pairs": pairs,
            "shard_id": shard_id,
        }
    except Exception as e:
        push_status(state="error", detail="", error=str(e), finished=True)
        _emit("error", batch=batch_id, msg=str(e).replace(" ", "_"))
        raise


def run_parallel(
    batch_id: str,
    *,
    workers: int = 10,
    skip_ingest: bool = True,
    poll_s: float = 2.0,
) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        raise FileNotFoundError(f"no batch {batch_id}")

    pairs = [int(p) for p in manifest["pairs"]]
    shards = split_pair_shards(pairs, workers)
    if not shards:
        raise ValueError(f"no pairs in batch {batch_id}")

    eval_runs.clear_shard_statuses(batch_id)
    started_at = int(time.time())
    eval_runs.write_status(
        batch_id,
        {
            "state": "running",
            "done": 0,
            "total": 0,
            "detail": f"spawning {len(shards)} workers",
            "error": None,
            "workers": len(shards),
            "started_at": started_at,
        },
    )

    cli = str(Path(__file__).resolve())
    batch_root = eval_runs.batch_dir(batch_id)
    batch_root.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []
    log_handles: list = []
    for i, shard_pairs in enumerate(shards):
        lo, hi = shard_pairs[0], shard_pairs[-1]
        pair_spec = (
            f"{lo}-{hi}"
            if shard_pairs == list(range(lo, hi + 1))
            else ",".join(str(p) for p in shard_pairs)
        )
        cmd = [
            sys.executable,
            cli,
            "run",
            batch_id,
            "--pairs",
            pair_spec,
            "--shard-id",
            str(i),
        ]
        if skip_ingest:
            cmd.append("--skip-ingest")
        log_path = batch_root / f"shard-{i}.log"
        log_fh = open(log_path, "w")
        log_handles.append(log_fh)
        _emit(
            "spawn",
            shard=i,
            pairs=f"{shard_pairs[0]}-{shard_pairs[-1]}",
            n=len(shard_pairs),
            log=str(log_path),
        )
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        )

    try:
        while True:
            alive = sum(1 for p in procs if p.poll() is None)
            for i, p in enumerate(procs):
                code = p.poll()
                if code is None:
                    continue
                st = eval_runs.read_shard_status(batch_id, i) or {}
                if st.get("state") in ("done", "error"):
                    continue
                if code == 0:
                    eval_runs.write_shard_status(
                        batch_id,
                        i,
                        {
                            "state": "done",
                            "done": int(st.get("done") or st.get("total") or 0),
                            "total": int(st.get("total") or 0),
                            "detail": st.get("detail") or "complete",
                            "error": None,
                            "pairs": st.get("pairs") or shards[i],
                            "started_at": st.get("started_at") or started_at,
                            "finished_at": int(time.time()),
                        },
                    )
                else:
                    eval_runs.write_shard_status(
                        batch_id,
                        i,
                        {
                            "state": "error",
                            "done": int(st.get("done") or 0),
                            "total": int(st.get("total") or 0),
                            "detail": st.get("detail") or "",
                            "error": f"shard {i} exited {code}",
                            "pairs": st.get("pairs") or shards[i],
                            "started_at": st.get("started_at") or started_at,
                            "finished_at": int(time.time()),
                        },
                    )

            agg = eval_runs.aggregate_status(batch_id)
            if "started_at" not in agg:
                agg["started_at"] = started_at
            eval_runs.write_status(batch_id, agg)

            if alive == 0:
                break
            time.sleep(max(0.2, float(poll_s)))

        codes = [p.wait() for p in procs]
        agg = eval_runs.aggregate_status(batch_id)
        if "started_at" not in agg:
            agg["started_at"] = started_at
        if any(c != 0 for c in codes) and agg.get("state") != "error":
            failed = [i for i, c in enumerate(codes) if c != 0]
            agg["state"] = "error"
            agg["error"] = f"shards failed: {failed}"
            agg["finished_at"] = int(time.time())
        elif agg.get("state") != "error":
            agg["state"] = "done"
            agg["finished_at"] = int(time.time())
        eval_runs.write_status(batch_id, agg)
        _emit(
            "parallel_done",
            batch=batch_id,
            workers=len(shards),
            done=agg.get("done"),
            total=agg.get("total"),
            state=agg.get("state"),
        )
        ok = agg.get("state") == "done" and all(c == 0 for c in codes)
        return {
            "ok": ok,
            "batch_id": batch_id,
            "workers": len(shards),
            "exit_codes": codes,
            "status": agg,
            "shards": [
                {"shard_id": i, "pairs": s, "lo": s[0], "hi": s[-1]}
                for i, s in enumerate(shards)
            ],
        }
    except Exception:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        for fh in log_handles:
            try:
                fh.close()
            except Exception:
                pass


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def cmd_list() -> None:
    _print_json({"batches": eval_runs.list_batches()})


def cmd_create(args: argparse.Namespace) -> None:
    pairs = parse_pairs_spec(args.pairs)
    config = eval_runs.default_config()
    if args.wendland_eps is not None:
        config["wendland_eps"] = float(args.wendland_eps)
    if args.bspline_grid is not None:
        config["bspline_grid"] = int(args.bspline_grid)
    if args.bspline_reg is not None:
        config["bspline_reg"] = float(args.bspline_reg)
    if args.force:
        config["force"] = True
    lams = [x.strip() for x in args.lams.split(",")] if args.lams else None
    estimators = (
        [x.strip() for x in args.estimators.split(",")] if args.estimators else None
    )
    try:
        man = eval_runs.create_batch(
            args.name,
            pairs,
            lams=lams,
            estimators=estimators,
            config=config,
            notes=args.notes,
            batch_id=args.id,
            dataset=args.dataset,
        )
    except FileExistsError as e:
        _print_json({"ok": False, "error": str(e)})
        sys.exit(1)
    _print_json({"ok": True, "manifest": man})


def cmd_run(args: argparse.Namespace) -> None:
    pair_ids = parse_pairs_spec(args.pairs) if args.pairs else None
    _print_json(
        run_batch(
            args.batch_id,
            pair_ids=pair_ids,
            skip_ingest=bool(args.skip_ingest),
            shard_id=args.shard_id,
        )
    )


def cmd_run_parallel(args: argparse.Namespace) -> None:
    result = run_parallel(
        args.batch_id,
        workers=int(args.workers),
        skip_ingest=bool(args.skip_ingest),
        poll_s=float(args.poll_s),
    )
    _print_json(result)
    if not result.get("ok"):
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--pairs", required=True, help="comma-separated or ranges, e.g. 0-9,15")
    c.add_argument("--id", default=None, help="optional batch id (slug)")
    c.add_argument("--lams", default=None, help="comma-separated, default all")
    c.add_argument("--estimators", default=None, help="comma-separated, default all")
    c.add_argument("--dataset", default="muromi", help="muromi|acrobat")
    c.add_argument("--wendland-eps", type=float, default=None)
    c.add_argument("--bspline-grid", type=int, default=None)
    c.add_argument("--bspline-reg", type=float, default=None)
    c.add_argument("--force", action="store_true")
    c.add_argument("--notes", default="")

    r = sub.add_parser("run")
    r.add_argument("batch_id")
    r.add_argument(
        "--pairs",
        default=None,
        help="subset of manifest pairs: 0-9 or 0,1,2 or 0-9,15",
    )
    r.add_argument(
        "--skip-ingest",
        action="store_true",
        help="skip acrobat ingest (use after a separate re-ingest)",
    )
    r.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="write status.shard-{id}.json instead of status.json",
    )

    rp = sub.add_parser("run-parallel")
    rp.add_argument("batch_id")
    rp.add_argument("--workers", type=int, default=10)
    rp.add_argument(
        "--skip-ingest",
        dest="skip_ingest",
        action="store_true",
        default=True,
        help="pass --skip-ingest to each shard (default)",
    )
    rp.add_argument(
        "--ingest",
        dest="skip_ingest",
        action="store_false",
        help="allow each shard to run ingest (not recommended)",
    )
    rp.add_argument("--poll-s", type=float, default=2.0)

    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "create":
        cmd_create(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "run-parallel":
        cmd_run_parallel(args)


if __name__ == "__main__":
    main()
