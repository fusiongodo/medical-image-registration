"""
Eval batch CLI: create / list / run LAM × field-estimator grids.

Usage:
  python setup/coarse_to_fine/eval_batch_cli.py list
  python setup/coarse_to_fine/eval_batch_cli.py create --name demo --pairs 0,1,4,16
  python setup/coarse_to_fine/eval_batch_cli.py run <batch_id>
  python setup/coarse_to_fine/eval_batch_cli.py run <batch_id> --pairs 0-9 --skip-ingest --shard-id 0
  python setup/coarse_to_fine/eval_batch_cli.py run-parallel <batch_id> --schedule resource --gpu-workers 3 --cpu-workers 7
  python setup/coarse_to_fine/eval_batch_cli.py run-parallel <batch_id> --schedule shards --workers 10 --skip-ingest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
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
from setup.coarse_to_fine.reg_branches import cache_path, normalize_estimator, normalize_lam
from setup.coarse_to_fine.run import _level_records, write_candidate_cache


def _canvas_ingest(ds_name: str, pairs: list[int], *, force: bool = False) -> dict | None:
    if ds_name == "acrobat":
        from setup.acrobat.ingest import ingest

        return ingest(unzip=True, pair_ids=pairs, force=force)
    if ds_name == "anhir":
        from setup.anhir.ingest import ingest

        return ingest(pair_ids=pairs, force=force)
    return None


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


def _lam_is_gpu(lam: str) -> bool:
    return normalize_lam(lam) != "fft"


def _regwsi_ready(pair_id: int, *, force: bool = False) -> bool:
    if force:
        return False
    from regWSI import paths as rpaths

    return (
        rpaths.displacement_field(pair_id).is_file()
        and datasets.rigid_path(pair_id).is_file()
    )


def _migrate_regwsi_rigids(pairs: list[int]) -> None:
    from regWSI.extract_rigid import ensure_regwsi_rigid

    for pair_id in pairs:
        if ensure_regwsi_rigid(pair_id):
            _emit("rigid_migrated", pair=pair_id)


def _lam_pending(
    batch_id: str,
    pair_id: int,
    lam: str,
    estimators: list[str],
    cell_cfg: dict,
    *,
    force: bool = False,
) -> bool:
    if force:
        return True
    return any(
        not eval_runs.cell_complete(batch_id, pair_id, lam, est, cell_cfg)
        for est in estimators
    )


def _batch_progress_counts(
    batch_id: str,
    pairs: list[int],
    lams: list[str],
    estimators: list[str],
    cell_cfg: dict,
    ds_name: str,
    *,
    force: bool = False,
) -> tuple[int, int]:
    total = len(pairs) * len(lams) * len(estimators)
    done = 0
    if datasets.uses_pair_tiffs(ds_name):
        total += len(pairs)
        for p in pairs:
            if _regwsi_ready(p, force=False) and not force:
                done += 1
    for p in pairs:
        for lam in lams:
            for est in estimators:
                if eval_runs.cell_complete(batch_id, p, lam, est, cell_cfg) and not force:
                    done += 1
    return done, total


def _emit(stage: str, **kv) -> None:
    parts = [f"stage={stage}"]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts), flush=True)


def _load_level_candidates(
    pair_id: int,
    level: int,
    lam: str,
    field_estimator: str,
    wendland_eps: float | None = None,
) -> list[Candidate]:
    path = cache_path(
        pair_id,
        level,
        lam,
        field_estimator=field_estimator,
        wendland_eps=wendland_eps,
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [candidate_from_dict(level, d) for d in payload.get("candidates", [])]


def _fit_pair_lam_estimator(
    pair_id: int,
    levels: list[int],
    lam: str,
    estimator: str,
    wendland_eps: float,
    bspline_grid: int,
    bspline_reg: float,
    force: bool,
) -> tuple[Field, dict]:
    import setup.coarse_to_fine.field as field_mod

    field_mod.WENDLAND_EPS = float(wendland_eps)
    field_mod.BSPLINE_GRID = int(bspline_grid)
    field_mod.BSPLINE_REG = float(bspline_reg)

    entries = annotations.load(pair_id)
    mask_entries = masks.load(pair_id)
    t_init = time.perf_counter()
    field = fit_field(
        annotations.to_anchors(entries, up_to_level=min(levels) - 1),
        field_estimator=estimator,
        wendland_epsilon=wendland_eps,
        bspline_grid=bspline_grid,
        bspline_reg=bspline_reg,
    )
    init_fit_s = time.perf_counter() - t_init

    total_kept = 0
    total_seen = 0
    level_gates: dict[str, dict] = {}
    level_times: dict[str, dict] = {}
    lam_s_total = 0.0
    fit_s_total = init_fit_s
    for level in sorted(levels):
        path = cache_path(
            pair_id,
            level,
            lam,
            field_estimator=estimator,
            wendland_eps=wendland_eps,
        )
        t_lam = time.perf_counter()
        if path.exists() and not force:
            _emit("cache_hit", pair=pair_id, lam=lam, estimator=estimator, level=level)
            cands = _load_level_candidates(
                pair_id, level, lam, estimator, wendland_eps=wendland_eps
            )
            cache_kind = "hit"
        else:
            _emit(
                "cache_compute",
                pair=pair_id,
                lam=lam,
                estimator=estimator,
                level=level,
            )
            records = _level_records(pair_id, level, field, lam=lam)
            write_candidate_cache(
                pair_id,
                level,
                records,
                lam=lam,
                field_estimator=estimator,
                levels=levels,
                wendland_eps=wendland_eps,
            )
            cands = [candidate_from_dict(level, d) for d in records]
            cache_kind = "compute"
        lam_s = time.perf_counter() - t_lam
        if not cands:
            level_times[str(level)] = {
                "lam_s": lam_s,
                "fit_s": 0.0,
                "cache": cache_kind,
                "n_cands": 0,
            }
            lam_s_total += lam_s
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
        t_fit = time.perf_counter()
        if keep >= 1.0:
            tau = float("inf")
        else:
            tau = tau_for_keep(human, fit_cands, keep, field_estimator=estimator)
        field, kept = fit_gated(
            human, fit_cands, tau, field_estimator=estimator
        )
        fit_s = time.perf_counter() - t_fit
        level_times[str(level)] = {
            "lam_s": lam_s,
            "fit_s": fit_s,
            "cache": cache_kind,
            "n_cands": len(fit_cands),
            "n_kept": len(kept),
        }
        lam_s_total += lam_s
        fit_s_total += fit_s
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
        "level_times": level_times,
        "lam_s": lam_s_total,
        "fit_s": fit_s_total,
        "init_fit_s": init_fit_s,
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
    identity = datasets.pair_fingerprint(pair_id, ds)
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
    lam_filter: list[str] | None = None,
    regwsi_only: bool = False,
    skip_regwsi: bool = False,
) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        raise FileNotFoundError(f"no batch {batch_id}")
    if regwsi_only and skip_regwsi:
        raise ValueError("regwsi_only and skip_regwsi are mutually exclusive")

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

    if datasets.uses_pair_tiffs(ds_name):
        _migrate_regwsi_rigids(pairs)

    lams = [normalize_lam(x) for x in manifest.get("lams") or []]
    if lam_filter is not None:
        want = {normalize_lam(x) for x in lam_filter}
        lams = [x for x in lams if x in want]
        if not lams and not regwsi_only:
            raise ValueError("lam filter matched no manifest lams")
    estimators = [normalize_estimator(x) for x in manifest.get("estimators") or []]
    cfg = {**eval_runs.default_config(), **(manifest.get("config") or {})}
    cell_cfg = eval_runs.cell_config(cfg)
    fp = eval_runs.config_fingerprint(cell_cfg)
    levels = sorted(int(x) for x in cell_cfg["levels"])
    force = bool(cfg.get("force")) or bool(manifest.get("config", {}).get("force"))
    bspline_grid = int(cell_cfg["bspline_grid"])
    bspline_reg = float(cell_cfg["bspline_reg"])

    do_regwsi = datasets.uses_pair_tiffs(ds_name) and not skip_regwsi
    do_cells = not regwsi_only
    jobs = (
        [(p, lam, est) for p in pairs for lam in lams for est in estimators]
        if do_cells
        else []
    )
    total = len(jobs) + (len(pairs) if do_regwsi else 0)
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
        detail="ingest" if (datasets.uses_pair_tiffs(ds_name) and not skip_ingest) else "start",
    )

    if datasets.uses_pair_tiffs(ds_name) and not skip_ingest:
        _emit("ingest", dataset=ds_name, pairs=len(pairs), done=0, total=total)
        ingest_result = _canvas_ingest(ds_name, pairs, force=False) or {}
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
        if do_regwsi:
            from regWSI import paths as rpaths
            from regWSI.register import register_pair

            for pair_id in pairs:
                detail = f"pair={pair_id} stage=regwsi"
                _emit("regwsi", pair=pair_id, done=done, total=total)
                push_status(state="running", detail=detail)
                df = rpaths.displacement_field(pair_id)
                rigid = datasets.rigid_path(pair_id)
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
            field, meta = _fit_pair_lam_estimator(
                pair_id,
                levels,
                lam,
                estimator,
                eval_runs.eps_for_lam(cell_cfg, lam),
                bspline_grid,
                bspline_reg,
                force=force,
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


def run_parallel_shards(
    batch_id: str,
    *,
    workers: int = 10,
    skip_ingest: bool = True,
    poll_s: float = 2.0,
    pair_ids: list[int] | None = None,
) -> dict:
    manifest = eval_runs.read_manifest(batch_id)
    if manifest is None:
        raise FileNotFoundError(f"no batch {batch_id}")

    manifest_pairs = [int(p) for p in manifest["pairs"]]
    allowed = set(manifest_pairs)
    if pair_ids is None:
        pairs = list(manifest_pairs)
    else:
        pairs = [int(p) for p in pair_ids if int(p) in allowed]
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


def run_parallel_resource(
    batch_id: str,
    *,
    gpu_workers: int = 3,
    cpu_workers: int = 7,
    skip_ingest: bool = True,
    poll_s: float = 2.0,
    pair_ids: list[int] | None = None,
) -> dict:
    """GPU queue: regWSI then superpoint; CPU queue: fft. Shared job deques."""
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
    if datasets.uses_pair_tiffs(ds_name):
        _migrate_regwsi_rigids(pairs)
    lams = [normalize_lam(x) for x in manifest.get("lams") or []]
    estimators = [normalize_estimator(x) for x in manifest.get("estimators") or []]
    cfg = {**eval_runs.default_config(), **(manifest.get("config") or {})}
    cell_cfg = eval_runs.cell_config(cfg)
    force = bool(cfg.get("force")) or bool(manifest.get("config", {}).get("force"))

    if not skip_ingest and datasets.uses_pair_tiffs(ds_name):
        _emit("ingest", dataset=ds_name, pairs=len(pairs))
        _canvas_ingest(ds_name, pairs, force=False)

    n_gpu = max(0, int(gpu_workers))
    n_cpu = max(0, int(cpu_workers))
    if n_gpu + n_cpu < 1:
        raise ValueError("need at least one gpu or cpu worker")

    gpu_regwsi: deque[dict] = deque()
    gpu_sp: deque[dict] = deque()
    cpu_fft: deque[dict] = deque()
    queued: set[tuple] = set()

    def _enqueue(job: dict) -> None:
        kind = job["kind"]
        if kind == "regwsi":
            key = ("regwsi", int(job["pair"]))
            if key in queued:
                return
            queued.add(key)
            gpu_regwsi.append(job)
        elif kind == "lam":
            lam = normalize_lam(job["lam"])
            key = ("lam", int(job["pair"]), lam)
            if key in queued:
                return
            queued.add(key)
            job = {**job, "lam": lam}
            if _lam_is_gpu(lam):
                gpu_sp.append(job)
            else:
                cpu_fft.append(job)

    def _seed_lam_jobs(pair_id: int) -> None:
        for lam in lams:
            if _lam_pending(
                batch_id, pair_id, lam, estimators, cell_cfg, force=force
            ):
                _enqueue({"kind": "lam", "pair": int(pair_id), "lam": lam})

    for pair_id in pairs:
        if datasets.uses_pair_tiffs(ds_name) and not _regwsi_ready(pair_id, force=force):
            _enqueue({"kind": "regwsi", "pair": int(pair_id)})
        else:
            _seed_lam_jobs(pair_id)

    eval_runs.clear_shard_statuses(batch_id)
    started_at = int(time.time())
    done0, total = _batch_progress_counts(
        batch_id, pairs, lams, estimators, cell_cfg, ds_name, force=force
    )
    eval_runs.write_status(
        batch_id,
        {
            "state": "running",
            "done": done0,
            "total": total,
            "detail": f"resource queue gpu={n_gpu} cpu={n_cpu}",
            "error": None,
            "workers": n_gpu + n_cpu,
            "schedule": "resource",
            "started_at": started_at,
        },
    )

    cli = str(Path(__file__).resolve())
    batch_root = eval_runs.batch_dir(batch_id)
    batch_root.mkdir(parents=True, exist_ok=True)

    slots: list[dict] = []
    for i in range(n_gpu):
        slots.append(
            {
                "role": "gpu",
                "gpu_index": i,
                "shard_id": i,
                "proc": None,
                "job": None,
                "log": None,
            }
        )
    for j in range(n_cpu):
        slots.append(
            {
                "role": "cpu",
                "gpu_index": None,
                "shard_id": n_gpu + j,
                "proc": None,
                "job": None,
                "log": None,
            }
        )

    def _pop_job(role: str) -> dict | None:
        if role == "gpu":
            if gpu_regwsi:
                return gpu_regwsi.popleft()
            if gpu_sp:
                return gpu_sp.popleft()
            return None
        if cpu_fft:
            return cpu_fft.popleft()
        return None

    def _spawn(slot: dict, job: dict) -> None:
        shard_id = int(slot["shard_id"])
        pair = int(job["pair"])
        cmd = [
            sys.executable,
            cli,
            "run",
            batch_id,
            "--pairs",
            str(pair),
            "--shard-id",
            str(shard_id),
            "--skip-ingest",
        ]
        if job["kind"] == "regwsi":
            cmd.append("--regwsi-only")
        else:
            cmd.extend(["--skip-regwsi", "--lams", job["lam"]])
        log_path = batch_root / f"shard-{shard_id}.log"
        log_fh = open(log_path, "a")
        slot["log"] = log_fh
        slot["job"] = job
        env = os.environ.copy()
        if slot["role"] == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(int(slot.get("gpu_index") or 0))
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        slot["proc"] = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        _emit(
            "spawn_job",
            role=slot["role"],
            shard=shard_id,
            kind=job["kind"],
            pair=pair,
            lam=job.get("lam", "-"),
            cuda=env.get("CUDA_VISIBLE_DEVICES", ""),
        )

    errors: list[str] = []
    try:
        while True:
            for slot in slots:
                proc = slot["proc"]
                if proc is None:
                    continue
                code = proc.poll()
                if code is None:
                    continue
                job = slot["job"] or {}
                if slot["log"] is not None:
                    try:
                        slot["log"].close()
                    except Exception:
                        pass
                slot["log"] = None
                slot["proc"] = None
                slot["job"] = None
                if code != 0:
                    errors.append(
                        f"{slot['role']}/shard-{slot['shard_id']} "
                        f"{job.get('kind')} pair={job.get('pair')} exit={code}"
                    )
                    _emit(
                        "job_error",
                        role=slot["role"],
                        pair=job.get("pair"),
                        kind=job.get("kind"),
                        code=code,
                    )
                else:
                    _emit(
                        "job_done",
                        role=slot["role"],
                        pair=job.get("pair"),
                        kind=job.get("kind"),
                        lam=job.get("lam", "-"),
                    )
                    if (
                        job.get("kind") == "regwsi"
                        and datasets.uses_pair_tiffs(ds_name)
                        and _regwsi_ready(int(job["pair"]), force=False)
                    ):
                        _seed_lam_jobs(int(job["pair"]))

            for slot in slots:
                if slot["proc"] is not None:
                    continue
                job = _pop_job(slot["role"])
                if job is None:
                    continue
                _spawn(slot, job)

            done, total = _batch_progress_counts(
                batch_id, pairs, lams, estimators, cell_cfg, ds_name, force=force
            )
            running = [
                s
                for s in slots
                if s["proc"] is not None and s["job"] is not None
            ]
            detail_parts = []
            for s in running[:4]:
                j = s["job"]
                bit = f"{s['role']}:{j['kind']} p={j['pair']}"
                if j.get("lam"):
                    bit += f" lam={j['lam']}"
                detail_parts.append(bit)
            qinfo = (
                f"q_regwsi={len(gpu_regwsi)} q_sp={len(gpu_sp)} q_fft={len(cpu_fft)}"
            )
            detail = "; ".join(detail_parts) if detail_parts else qinfo
            if detail_parts:
                detail = f"{detail} · {qinfo}"
            state = "error" if errors and not running and not (
                gpu_regwsi or gpu_sp or cpu_fft
            ) else "running"
            eval_runs.write_status(
                batch_id,
                {
                    "state": state,
                    "done": done,
                    "total": total,
                    "detail": detail,
                    "error": errors[0] if errors else None,
                    "workers": n_gpu + n_cpu,
                    "schedule": "resource",
                    "started_at": started_at,
                },
            )

            pending_q = bool(gpu_regwsi or gpu_sp or cpu_fft)
            if not running and not pending_q:
                break
            time.sleep(max(0.2, float(poll_s)))

        done, total = _batch_progress_counts(
            batch_id, pairs, lams, estimators, cell_cfg, ds_name, force=force
        )
        ok = not errors and done >= total
        eval_runs.write_status(
            batch_id,
            {
                "state": "done" if ok else "error",
                "done": done,
                "total": total,
                "detail": "complete" if ok else "resource queue finished with errors",
                "error": errors[0] if errors else None,
                "workers": n_gpu + n_cpu,
                "schedule": "resource",
                "started_at": started_at,
                "finished_at": int(time.time()),
            },
        )
        _emit(
            "parallel_done",
            batch=batch_id,
            schedule="resource",
            gpu=n_gpu,
            cpu=n_cpu,
            done=done,
            total=total,
            errors=len(errors),
        )
        return {
            "ok": ok,
            "batch_id": batch_id,
            "schedule": "resource",
            "gpu_workers": n_gpu,
            "cpu_workers": n_cpu,
            "done": done,
            "total": total,
            "errors": errors,
        }
    except Exception:
        for slot in slots:
            proc = slot["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()
            if slot["log"] is not None:
                try:
                    slot["log"].close()
                except Exception:
                    pass
        raise


def run_parallel(
    batch_id: str,
    *,
    schedule: str = "resource",
    workers: int = 10,
    gpu_workers: int = 3,
    cpu_workers: int = 7,
    skip_ingest: bool = True,
    poll_s: float = 2.0,
    pair_ids: list[int] | None = None,
) -> dict:
    sched = (schedule or "resource").strip().lower()
    if sched == "shards":
        return run_parallel_shards(
            batch_id,
            workers=workers,
            skip_ingest=skip_ingest,
            poll_s=poll_s,
            pair_ids=pair_ids,
        )
    if sched != "resource":
        raise ValueError(f"unknown schedule {schedule!r}; use resource|shards")
    return run_parallel_resource(
        batch_id,
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        skip_ingest=skip_ingest,
        poll_s=poll_s,
        pair_ids=pair_ids,
    )


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def cmd_list() -> None:
    _print_json({"batches": eval_runs.list_batches()})


def parse_eps_by_lam(spec: str | None) -> dict[str, float] | None:
    if spec is None or not str(spec).strip():
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected lam=eps in {spec!r}, got {part!r}")
        lam, raw = part.split("=", 1)
        out[normalize_lam(lam.strip())] = float(raw.strip())
    return out or None


def cmd_create(args: argparse.Namespace) -> None:
    pairs = parse_pairs_spec(args.pairs)
    config = eval_runs.default_config()
    if args.wendland_eps is not None:
        config["wendland_eps"] = float(args.wendland_eps)
    by_lam = parse_eps_by_lam(getattr(args, "wendland_eps_by_lam", None))
    if by_lam:
        config["wendland_eps_by_lam"] = by_lam
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
    lam_filter = None
    if args.lams:
        lam_filter = [x.strip() for x in args.lams.split(",") if x.strip()]
    _print_json(
        run_batch(
            args.batch_id,
            pair_ids=pair_ids,
            skip_ingest=bool(args.skip_ingest),
            shard_id=args.shard_id,
            lam_filter=lam_filter,
            regwsi_only=bool(args.regwsi_only),
            skip_regwsi=bool(args.skip_regwsi),
        )
    )


def cmd_run_parallel(args: argparse.Namespace) -> None:
    pair_ids = parse_pairs_spec(args.pairs) if args.pairs else None
    result = run_parallel(
        args.batch_id,
        schedule=str(args.schedule),
        workers=int(args.workers),
        gpu_workers=int(args.gpu_workers),
        cpu_workers=int(args.cpu_workers),
        skip_ingest=bool(args.skip_ingest),
        poll_s=float(args.poll_s),
        pair_ids=pair_ids,
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
    c.add_argument("--dataset", default="muromi", help="muromi|acrobat|anhir")
    c.add_argument("--wendland-eps", type=float, default=None)
    c.add_argument(
        "--wendland-eps-by-lam",
        default=None,
        help="per-LAM compact support, e.g. fft=0.2,superpoint_glue=0.1",
    )
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
    r.add_argument(
        "--lams",
        default=None,
        help="comma-separated lam subset for this run (e.g. fft)",
    )
    r.add_argument(
        "--regwsi-only",
        action="store_true",
        help="only run acrobat regWSI for selected pairs",
    )
    r.add_argument(
        "--skip-regwsi",
        action="store_true",
        help="skip regWSI; only LAM×field cells",
    )

    rp = sub.add_parser("run-parallel")
    rp.add_argument("batch_id")
    rp.add_argument(
        "--schedule",
        default="resource",
        choices=("resource", "shards"),
        help="resource=GPU/CPU job queues (default); shards=pair slices",
    )
    rp.add_argument(
        "--workers",
        type=int,
        default=10,
        help="pair-shard worker count when --schedule shards",
    )
    rp.add_argument(
        "--gpu-workers",
        type=int,
        default=3,
        help="resource schedule: workers for regWSI + superpoint (default 3)",
    )
    rp.add_argument(
        "--cpu-workers",
        type=int,
        default=7,
        help="resource schedule: workers for fft (default 7)",
    )
    rp.add_argument(
        "--skip-ingest",
        dest="skip_ingest",
        action="store_true",
        default=True,
        help="skip acrobat ingest (default)",
    )
    rp.add_argument(
        "--ingest",
        dest="skip_ingest",
        action="store_false",
        help="allow ingest before queue (not recommended)",
    )
    rp.add_argument("--poll-s", type=float, default=2.0)
    rp.add_argument(
        "--pairs",
        default=None,
        help="subset of manifest pairs: 0-9 or 0,1,2 or 0-9,15",
    )

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
