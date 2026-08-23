"""
SP rotational-invariance training CLI.

  python setup/coarse_to_fine/sp_rot_train_cli.py create --name run1
  python setup/coarse_to_fine/sp_rot_train_cli.py run <run_id>
  python setup/coarse_to_fine/sp_rot_train_cli.py pause <run_id>
  python setup/coarse_to_fine/sp_rot_train_cli.py resume <run_id>
  python setup/coarse_to_fine/sp_rot_train_cli.py status <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "introducing_superpoint"))

from setup.coarse_to_fine import sp_rot_bench as bench
from setup.coarse_to_fine import sp_rot_train as store
from setup.coarse_to_fine import sp_rot_train_data as data
from setup.coarse_to_fine import sp_rot_train_eval as beval


def _safe_print(*args, **kwargs) -> None:
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        pass


def emit(obj) -> None:
    _safe_print(bench.dumps(obj, indent=2), flush=True)


def cmd_list(_: argparse.Namespace) -> None:
    emit({"runs": store.list_runs()})


def cmd_create(args: argparse.Namespace) -> None:
    cfg = {}
    if args.pairs:
        cfg["pairs"] = [int(x) for x in args.pairs.split(",") if x.strip() != ""]
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.max_epochs is not None:
        cfg["max_epochs"] = int(args.max_epochs)
    if args.ckpt_every_epochs is not None:
        cfg["ckpt_every_epochs"] = int(args.ckpt_every_epochs)
    if args.eval_every_epochs is not None:
        cfg["eval_every_epochs"] = int(args.eval_every_epochs)
    if args.log_every is not None:
        cfg["log_every"] = int(args.log_every)
    if args.split_seed is not None:
        cfg["split_seed"] = int(args.split_seed)
    if args.eval_max_tiles is not None:
        cfg["eval_max_tiles"] = int(args.eval_max_tiles)
    if args.run_baseline:
        cfg["skip_baseline"] = False
    for name in (
        "depth",
        "max_steps",
        "eval_every_steps",
        "ckpt_every_steps",
        "eval_step_tiles",
        "desc_max_cells",
        "desc_lambda",
        "kp_loss",
        "num_workers",
        "gt_nms_dist",
        "gt_conf_thresh",
    ):
        v = getattr(args, name, None)
        if v is not None:
            cfg[name] = v
    if args.eval_angles:
        cfg["eval_angles"] = [float(x) for x in args.eval_angles.split(",") if x.strip()]
    if args.frozen_gt:
        cfg["frozen_gt"] = True
    try:
        man = store.create_run(args.name, config=cfg, run_id=args.id)
        emit({"ok": True, "run": man})
    except FileExistsError as e:
        emit({"ok": False, "error": f"run already exists: {e}"})
        sys.exit(1)
    except Exception as e:
        emit({"ok": False, "error": str(e)})
        sys.exit(1)


def cmd_pause(args: argparse.Namespace) -> None:
    store.pause_flag(args.run_id).write_text("1")
    store.update_status(args.run_id, state="pause_requested", detail="pause requested")
    emit({"ok": True})


def cmd_stop(args: argparse.Namespace) -> None:
    store.stop_flag(args.run_id).write_text("1")
    store.update_status(args.run_id, state="stop_requested", detail="stop requested")
    emit({"ok": True})


def cmd_status(args: argparse.Namespace) -> None:
    emit(
        {
            "config": store.load_config(args.run_id),
            "status": store.read_json(store.status_path(args.run_id)),
            "split": {
                k: store.load_split(args.run_id).get(k)
                for k in ("n_total", "n_train", "n_val", "n_test", "seed", "ratios")
            },
        }
    )


def _tail_jsonl(path: Path, n: int = 50) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    chunk = lines if n is None or int(n) <= 0 else lines[-int(n) :]
    out = []
    for line in chunk:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def cmd_logs(args: argparse.Namespace) -> None:
    emit(
        {
            "loss": _tail_jsonl(store.loss_log_path(args.run_id), args.n),
            "eval": _tail_jsonl(store.eval_log_path(args.run_id), args.n),
            "epoch": _tail_jsonl(store.epoch_log_path(args.run_id), args.n),
        }
    )


def _build_model(cfg: dict, device: torch.device):
    """Detector settings here are the *eval* ones: in-training eval extracts with this
    model's own nms_radius/detection_threshold, so they must match sp_rot_train_eval.
    GT density is a separate knob owned by the frozen teacher in sp_rot_gt_cache."""
    from training import build_model

    return build_model(
        cfg.get("init_weights"),
        device=device,
        nms_radius=int(cfg["sp_nms_dist"]),
        detection_threshold=float(cfg.get("eval_conf_thresh") or 0.015),
        max_num_keypoints=cfg.get("gt_max_kpts"),
    )


def _make_loader(tiles: list[dict], cfg: dict, *, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = data.RotWarpDataset(
        tiles,
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
    )
    nw = int(cfg.get("num_workers") or 0)
    kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": nw,
        "collate_fn": data.collate_rot,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
    }
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def _matcher_eval(
    model,
    tiles: list[dict],
    angles: list[float],
    cfg: dict,
    device,
    *,
    step: int,
    eval_log_path=None,
    tag: str = "eval",
) -> dict:
    """Dual-matcher eval condensed to one line per matcher: per-angle match counts + passing angles."""
    from setup.coarse_to_fine import sp_rot_train_eval as beval

    ev = beval.evaluate_tile_matchers(
        model,
        tiles,
        angles=angles,
        device=device,
        extract_resize=int(cfg["extract_resize"]),
        nms=int(cfg["sp_nms_dist"]),
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
        dataset=str(cfg.get("dataset") or "muromi"),
    )
    compact = {"step": step, "tag": tag, "n_tiles": len(tiles)}
    for kind in ("nn", "lg"):
        part = ev[kind]
        by_angle = part.get("by_angle") or {}
        matches = {}
        for c in part.get("cells") or []:
            matches.setdefault(f"{float(c['angle']):g}", []).append(int(c.get("n_matches") or 0))
        compact[kind] = {
            "n_pass": part.get("n_pass"),
            "n_total": part.get("n_total"),
            "pass_rate": part.get("pass_rate"),
            "n_error": part.get("n_error"),
            "by_angle": by_angle,
            "matches_mean": {k: round(sum(v) / len(v), 1) for k, v in matches.items()},
        }
        counts = ",".join(f"{k}:{v}" for k, v in compact[kind]["matches_mean"].items())
        ok = ",".join(
            f"{float(a):g}:{(r or {}).get('pass_rate') or 0:.2f}"
            for a, r in sorted(by_angle.items(), key=lambda kv: float(kv[0]))
        )
        _safe_print(
            f"{tag} step={step} matcher={kind} k={part['n_pass']}/{part['n_total']} "
            f"matches[{counts}] ok[{ok}]",
            flush=True,
        )
    if eval_log_path is not None:
        store.append_jsonl(eval_log_path, compact)
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return compact


def _run_eval(run_id: str, model, device, cfg: dict, tiles: list[dict], *, kind: str, step: int, epoch: int) -> dict:
    max_tiles = int(cfg.get("eval_max_tiles") or 0) or None
    store.update_status(
        run_id,
        state="running",
        step=step,
        epoch=epoch,
        detail=f"{kind} eval starting (≤{max_tiles or len(tiles)} tiles)…",
    )
    _safe_print(f"{kind} eval tiles={len(tiles)} max={max_tiles}…", flush=True)

    def on_progress(done: int, total: int, n_pass: int) -> None:
        store.update_status(
            run_id,
            state="running",
            step=step,
            epoch=epoch,
            detail=f"{kind} eval {done}/{total} cells pass={n_pass}",
        )

    ev = beval.evaluate_tiles(
        None,
        tiles,
        angles=[float(a) for a in cfg.get("eval_angles") or [0, 90, 180, 270]],
        extract_resize=int(cfg["extract_resize"]),
        nms=int(cfg["sp_nms_dist"]),
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
        max_tiles=max_tiles,
        split_seed=int(cfg.get("split_seed") or 0),
        dataset=str(cfg.get("dataset") or "muromi"),
        model=model,
        device=device,
        on_progress=on_progress,
    )
    ev.pop("cells", None)
    ev["kind"] = kind
    ev["step"] = step
    ev["epoch"] = epoch
    store.append_jsonl(store.eval_log_path(run_id), ev)
    store.update_status(
        run_id,
        last_eval=ev,
        detail=f"{kind} pass={ev.get('pass_rate')} ({ev.get('n_pass')}/{ev.get('n_total')})",
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _safe_print(f"{kind} pass_rate={ev.get('pass_rate')} ({ev.get('n_pass')}/{ev.get('n_total')})", flush=True)
    return ev


def _run_loop(run_id: str, *, resume: bool) -> None:
    cfg = store.load_config(run_id)
    split = store.load_split(run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, device)
    step = 0
    completed_epochs = 0
    st = store.read_json(store.status_path(run_id)) or {}
    latest = store.ckpt_dir(run_id) / "latest.pt"
    if resume and latest.is_file():
        store.load_checkpoint(model, latest, device)
        step = int(st.get("step") or 0)
        completed_epochs = int(st.get("epoch") or 0)
        meta = store.read_json(store.ckpt_dir(run_id) / "latest.json") or {}
        if meta.get("step") is not None:
            step = int(meta["step"])
        if meta.get("epoch") is not None:
            completed_epochs = int(meta["epoch"])

    for flag in (store.pause_flag(run_id), store.stop_flag(run_id)):
        if flag.is_file():
            flag.unlink()

    train_tiles = list(split["train"])
    val_tiles = list(split["val"])
    test_tiles = list(split["test"])
    train_loader = _make_loader(train_tiles, cfg, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_tiles, cfg, shuffle=False, drop_last=False)
    gt_cache = None
    if bool(cfg.get("frozen_gt")):
        from setup.coarse_to_fine import sp_rot_gt_cache as gtc

        gt_cache = gtc.GtCache(cfg, device)
        _safe_print(f"frozen GT cache at {gt_cache.dir}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    max_epochs = int(cfg["max_epochs"])
    store.update_status(
        run_id, state="running", step=step, epoch=completed_epochs, detail="start", error=None
    )

    if completed_epochs == 0 and step == 0 and not bool(cfg.get("skip_baseline")):
        _run_eval(run_id, model, device, cfg, val_tiles, kind="baseline", step=0, epoch=0)

    max_steps = int(cfg.get("max_steps") or 0)
    while completed_epochs < max_epochs and not (max_steps and step >= max_steps):
        cfg = store.load_config(run_id)
        max_epochs = int(cfg["max_epochs"])
        max_steps = int(cfg.get("max_steps") or 0)
        for g in opt.param_groups:
            g["lr"] = float(cfg["lr"])

        epoch = completed_epochs + 1
        t0 = time.time()
        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            if store.stop_flag(run_id).is_file():
                store.save_checkpoint(run_id, model, step, completed_epochs)
                if gt_cache is not None:
                    gt_cache.flush()
                store.update_status(
                    run_id, state="stopped", step=step, epoch=completed_epochs, detail="stopped"
                )
                emit({"ok": True, "state": "stopped", "step": step, "epoch": completed_epochs})
                return
            if store.pause_flag(run_id).is_file():
                store.save_checkpoint(run_id, model, step, completed_epochs)
                if gt_cache is not None:
                    gt_cache.flush()
                store.pause_flag(run_id).unlink(missing_ok=True)
                store.update_status(
                    run_id, state="paused", step=step, epoch=completed_epochs, detail="paused"
                )
                emit({"ok": True, "state": "paused", "step": step, "epoch": completed_epochs})
                return

            cfg = store.load_config(run_id)
            for g in opt.param_groups:
                g["lr"] = float(cfg["lr"])
            metrics = store.train_step(
                model,
                batch,
                opt,
                device,
                cfg,
                gt_base=gt_cache.get(batch) if gt_cache is not None else None,
            )
            step += 1
            train_loss_sum += metrics["loss_total"]
            train_n += 1
            store.update_status(
                run_id,
                state="running",
                step=step,
                epoch=completed_epochs,
                detail=f"epoch {epoch}/{max_epochs} loss={metrics['loss_total']:.4f}",
            )
            log_every = max(1, int(cfg.get("log_every") or 50))
            if step % log_every == 0:
                row = {
                    **metrics,
                    "step": step,
                    "epoch": epoch,
                    "lr": float(cfg["lr"]),
                    "ts": int(time.time()),
                }
                store.append_jsonl(store.loss_log_path(run_id), row)
                _safe_print(
                    f"step={step} loss={metrics['loss_total']:.4f} "
                    f"kp={metrics['loss_kp']:.4f} ce_kp={metrics['loss_fn']:.4f} "
                    f"ce_dust={metrics['loss_fp']:.6f} desc={metrics['loss_desc']:.4f} "
                    f"theta={metrics['theta_mean']:.0f}",
                    flush=True,
                )

            eval_every = int(cfg.get("eval_every_steps") or 0)
            if eval_every > 0 and step % eval_every == 0:
                _matcher_eval(
                    model,
                    val_tiles[: int(cfg.get("eval_step_tiles") or 2)],
                    [float(a) for a in cfg["eval_angles"]],
                    cfg,
                    device,
                    step=step,
                    eval_log_path=store.eval_log_path(run_id),
                    tag="eval",
                )

            ckpt_every = int(cfg.get("ckpt_every_steps") or 0)
            if ckpt_every > 0 and step % ckpt_every == 0:
                store.save_checkpoint(run_id, model, step, completed_epochs)
                if gt_cache is not None:
                    gt_cache.flush()
                    _safe_print(f"gt_cache {gt_cache.stats()}", flush=True)

            if max_steps and step >= max_steps:
                break

        epoch_s = time.time() - t0
        val_m = store.eval_loss(model, val_loader, device, cfg, gt_cache=gt_cache)
        completed_epochs = epoch
        store.save_checkpoint(run_id, model, step, completed_epochs)
        epoch_row = {
            "epoch": epoch,
            "step": step,
            "epoch_s": epoch_s,
            "train_loss": (train_loss_sum / train_n) if train_n else None,
            "val_loss": val_m.get("loss_total"),
            "val_loss_kp": val_m.get("loss_kp"),
            "val_loss_desc": val_m.get("loss_desc"),
            "ts": int(time.time()),
        }
        store.append_jsonl(store.epoch_log_path(run_id), epoch_row)
        store.update_status(
            run_id,
            step=step,
            epoch=completed_epochs,
            last_epoch_s=epoch_s,
            detail=f"epoch {epoch} done in {epoch_s:.1f}s val_loss={val_m.get('loss_total')}",
            error=None,
        )
        _safe_print(
            f"epoch={epoch}/{max_epochs} step={step} epoch_s={epoch_s:.1f} "
            f"train_loss={epoch_row['train_loss']} val_loss={epoch_row['val_loss']}",
            flush=True,
        )

        ckpt_every = max(1, int(cfg.get("ckpt_every_epochs") or 1))
        if epoch % ckpt_every == 0:
            _safe_print(f"ckpt epoch={epoch} step={step}", flush=True)

        eval_every = max(1, int(cfg.get("eval_every_epochs") or 5))
        if epoch % eval_every == 0:
            _run_eval(
                run_id, model, device, cfg, val_tiles, kind="val", step=step, epoch=epoch
            )

    store.save_checkpoint(run_id, model, step, completed_epochs)
    if gt_cache is not None:
        gt_cache.flush()
        _safe_print(f"gt_cache final {gt_cache.stats()}", flush=True)
    ev = _run_eval(
        run_id, model, device, cfg, test_tiles, kind="test", step=step, epoch=completed_epochs
    )
    store.update_status(
        run_id,
        state="done",
        step=step,
        epoch=completed_epochs,
        last_eval=ev,
        detail="finished",
    )
    emit(
        {
            "ok": True,
            "state": "done",
            "step": step,
            "epoch": completed_epochs,
            "last_eval": ev,
        }
    )


def cmd_run(args: argparse.Namespace) -> None:
    try:
        _run_loop(args.run_id, resume=False)
    except BrokenPipeError:
        emit({"ok": False, "error": "broken pipe (ignored)"})
    except Exception as e:
        store.update_status(args.run_id, state="error", error=str(e), detail="error")
        emit({"ok": False, "error": str(e)})
        raise


def cmd_resume(args: argparse.Namespace) -> None:
    try:
        _run_loop(args.run_id, resume=True)
    except BrokenPipeError:
        emit({"ok": False, "error": "broken pipe (ignored)"})
    except Exception as e:
        store.update_status(args.run_id, state="error", error=str(e), detail="error")
        emit({"ok": False, "error": str(e)})
        raise


def cmd_overfit(args: argparse.Namespace) -> None:
    import numpy as np
    from setup import datasets as ds

    run_id = args.run_id
    cfg = store.load_config(run_id)
    desc_max = int(args.desc_max_cells) if args.desc_max_cells is not None else 576
    cfg = {**cfg, "desc_max_cells": desc_max}
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if getattr(args, "desc_lambda", None) is not None:
        cfg["desc_lambda"] = float(args.desc_lambda)
    if getattr(args, "src_size", None) is not None:
        cfg["src_size"] = int(args.src_size)
    if getattr(args, "out_size", None) is not None:
        cfg["out_size"] = int(args.out_size)
    ds.set_active_dataset(cfg.get("dataset") or "muromi")

    split = store.load_split(run_id)
    train_tiles = list(split.get("train") or [])
    if not train_tiles:
        raise RuntimeError(f"no train tiles in split for {run_id}")

    n_tiles = max(1, int(args.n_tiles))
    if args.pair is not None and args.loc:
        matched = [
            t
            for t in train_tiles
            if int(t["pair_id"]) == int(args.pair) and str(t.get("loc")) == str(args.loc)
        ]
        if not matched:
            raise RuntimeError(f"tile pair={args.pair} loc={args.loc} not in train split")
        tiles = matched[:n_tiles]
    elif args.pair is not None:
        matched = [t for t in train_tiles if int(t["pair_id"]) == int(args.pair)]
        if not matched:
            raise RuntimeError(f"no train tile for pair={args.pair}")
        tiles = matched[:n_tiles]
    else:
        tiles = train_tiles[:n_tiles]
    if len(tiles) < n_tiles:
        raise RuntimeError(f"need {n_tiles} tiles, found {len(tiles)}")

    if args.angles:
        angles = [float(x) for x in str(args.angles).split(",") if x.strip() != ""]
    else:
        angles = [float(args.theta)]
    if not angles:
        raise RuntimeError("no angles to run")

    side = str(args.side)
    steps = int(args.steps)
    log_every = max(1, int(args.log_every))
    kp_loss = str(getattr(args, "kp_loss", "") or "matching")
    cfg = {**cfg, "kp_loss": kp_loss}
    exp_dir = store.TRAIN_ROOT / str(getattr(args, "exp_subdir", "") or "_overfit")
    exp_dir.mkdir(parents=True, exist_ok=True)
    experiments_path = exp_dir / "experiments.jsonl"

    import crop_core

    page_cache: dict[tuple[int, str], np.ndarray] = {}
    for tile in tiles:
        key = (int(tile["pair_id"]), side)
        if key not in page_cache:
            page = crop_core.whole_gray(key[0], side, int(cfg["preview_level"]))
            if page is None:
                raise RuntimeError(f"missing page pair={key[0]} side={side}")
            page_cache[key] = page

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_product = (not bool(args.each_tile)) and len(angles) > 1

    if use_product:
        specs = [(tile, float(th)) for tile in tiles for th in angles]
        batch_size = max(1, int(args.batch_size or min(10, len(specs))))
        _safe_print(
            f"=== overfit PRODUCT one model: {len(tiles)} tiles × {len(angles)} angles "
            f"= {len(specs)} samples, batch_size={batch_size}, lr={cfg['lr']} ===",
            flush=True,
        )

        samples = []
        for tile, theta in specs:
            key = (int(tile["pair_id"]), side)
            base, warped, valid, H = data.make_warp_pair(
                page_cache[key],
                tile,
                depth=int(cfg["depth"]),
                preview_level=int(cfg["preview_level"]),
                src_size=int(cfg["src_size"]),
                out_size=int(cfg["out_size"]),
                theta_deg=theta,
            )
            samples.append(
                {
                    "image": torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0),
                    "warped": torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0),
                    "valid_mask": torch.from_numpy((valid > 127).astype(np.float32)),
                    "homography": torch.from_numpy(H),
                    "theta_deg": theta,
                    "pair_id": int(tile["pair_id"]),
                    "loc": tile.get("loc"),
                    "side": side,
                }
            )

        model = _build_model(cfg, device)
        with torch.no_grad():
            for s in samples:
                img = s["image"].unsqueeze(0).to(device)
                Ht = s["homography"].unsqueeze(0).to(device)
                pseudo = store.detect_pseudo_gt(model, img, device)
                s["gt_base"] = [
                    g.detach().cpu()
                    for g in store.filter_gt_in_frame(pseudo, int(cfg["out_size"]))
                ]
                s["gt_warp"] = [
                    g.detach().cpu()
                    for g in store.filter_gt_in_frame(
                        store.warp_gt_points(pseudo, Ht), int(cfg["out_size"])
                    )
                ]
        n_gt = int(sum(int(g.shape[0]) for s in samples for g in s["gt_base"]))
        if n_gt == 0:
            raise RuntimeError("frozen pseudo-GT empty for product set")
        if args.init_weights:
            store.load_checkpoint(model, Path(args.init_weights), device)
            _safe_print(f"overfit product loaded weights {args.init_weights}", flush=True)

        def _gather(idxs: list[int]):
            batch = {
                "image": torch.stack([samples[i]["image"] for i in idxs], dim=0),
                "warped": torch.stack([samples[i]["warped"] for i in idxs], dim=0),
                "valid_mask": torch.stack([samples[i]["valid_mask"] for i in idxs], dim=0),
                "homography": torch.stack([samples[i]["homography"] for i in idxs], dim=0),
                "theta_deg": [samples[i]["theta_deg"] for i in idxs],
                "pair_id": [samples[i]["pair_id"] for i in idxs],
                "side": [samples[i]["side"] for i in idxs],
            }
            gt_base = [samples[i]["gt_base"][0].to(device) for i in idxs]
            gt_warp = [samples[i]["gt_warp"][0].to(device) for i in idxs]
            return batch, gt_base, gt_warp

        def _mean_kp() -> float:
            model.eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for start in range(0, len(samples), batch_size):
                    idxs = list(range(start, min(start + batch_size, len(samples))))
                    batch, gt_base, gt_warp = _gather(idxs)
                    parts = store._forward_losses(
                        model, batch, device, cfg, gt_base=gt_base, gt_warp=gt_warp
                    )
                    tot += float(parts["loss_kp"].detach().cpu())
                    n += 1
            model.train()
            return tot / max(1, n)

        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        eval_every = max(0, int(getattr(args, "eval_every", 0) or 0))
        stop_on_collapse = not bool(getattr(args, "no_collapse_stop", False))
        start_step = max(0, int(getattr(args, "start_step", 0) or 0))
        tag = str(getattr(args, "tag", "") or "").strip()
        stem = (
            f"overfit_product_t{len(tiles)}_a{len(angles)}"
            f"_b{batch_size}_d{int(cfg['desc_max_cells'])}"
        )
        if tag:
            stem = f"{stem}_{tag}"
        ckpt_path = exp_dir / f"{stem}.pt"
        ckpt_every = max(0, int(getattr(args, "ckpt_every", 0) or 0))
        tile_image = str(args.tile_image) if getattr(args, "tile_image", None) else None
        log_path = None
        eval_log_path = None
        if args.log:
            log_path = exp_dir / f"{stem}.jsonl"
            eval_log_path = exp_dir / f"{stem}_eval.jsonl"
            if log_path.is_file():
                log_path.unlink()
            if eval_log_path.is_file():
                eval_log_path.unlink()
            store.append_jsonl(
                log_path,
                {
                    "step": int(start_step),
                    "event": "start",
                    "tile_image": tile_image,
                    "tiles": [{"pair_id": int(t["pair_id"]), "loc": t.get("loc")} for t in tiles],
                    "ckpt_path": str(ckpt_path),
                    "ckpt_every": ckpt_every,
                    "start_step": int(start_step),
                    "steps": int(steps),
                    "init_weights": str(args.init_weights) if args.init_weights else None,
                    "kp_loss": kp_loss,
                    "src_size": int(cfg["src_size"]),
                    "out_size": int(cfg["out_size"]),
                },
            )
        _safe_print(
            f"overfit product tile_image={tile_image} ckpt_every={ckpt_every} "
            f"src_size={cfg['src_size']} out_size={cfg['out_size']} "
            f"same_fov={int(cfg['src_size'])==int(cfg['out_size'])} ckpt={ckpt_path}",
            flush=True,
        )

        def _save_latest() -> None:
            torch.save(model.state_dict(), ckpt_path)
            _safe_print(f"overfit product saved {ckpt_path}", flush=True)

        def _overfit_eval(step: int) -> dict:
            ev = beval.evaluate_tile_matchers(
                model,
                tiles,
                angles=angles,
                device=device,
                extract_resize=int(cfg["extract_resize"]),
                nms=int(cfg["sp_nms_dist"]),
                depth=int(cfg["depth"]),
                preview_level=int(cfg["preview_level"]),
                src_size=int(cfg["src_size"]),
                out_size=int(cfg["out_size"]),
                dataset=str(cfg.get("dataset") or "muromi"),
            )
            compact = {"step": step}
            for kind in ("nn", "lg"):
                part = ev[kind]
                compact[kind] = {
                    "n_pass": part.get("n_pass"),
                    "n_total": part.get("n_total"),
                    "pass_rate": part.get("pass_rate"),
                    "n_error": part.get("n_error"),
                    "by_angle": part.get("by_angle"),
                    "cells": [
                        {
                            "angle": c.get("angle"),
                            "auto_pass": c.get("auto_pass"),
                            "rot_err_deg": c.get("rot_err_deg"),
                            "trans_err_rel": c.get("trans_err_rel"),
                            "n_matches": c.get("n_matches"),
                            "n_inliers": c.get("n_inliers"),
                            "n_kp0": c.get("n_kp0"),
                            "n_kp1": c.get("n_kp1"),
                            "error": c.get("error"),
                        }
                        for c in part.get("cells") or []
                    ],
                }
            for kind in ("nn", "lg"):
                part = compact[kind]
                counts = ",".join(
                    f"{float(c['angle']):g}:{c.get('n_matches')}" for c in part["cells"]
                )
                passed = sorted(
                    float(c["angle"]) for c in part["cells"] if c.get("auto_pass")
                )
                _safe_print(
                    f"overfit product eval step={step} matcher={kind} "
                    f"k={part['n_pass']}/{part['n_total']} "
                    f"matches[{counts}] ok_angles={passed}",
                    flush=True,
                )
            if eval_log_path is not None:
                store.append_jsonl(eval_log_path, compact)
            model.train()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return compact

        kp0 = _mean_kp()
        _safe_print(
            f"overfit product step{start_step}_mean_kp={kp0:.6f} n_gt={n_gt} "
            f"start_step={start_step} steps={steps}",
            flush=True,
        )
        step0 = {"step": int(start_step), "loss_kp": kp0, "loss_total": None, "loss_desc": None}
        evals = []
        if eval_every > 0:
            evals.append(_overfit_eval(int(start_step)))
        first_collapse_step = None
        step_n = None
        history = []
        order = list(range(len(samples)))
        cursor = 0
        for step in range(start_step + 1, steps + 1):
            if cursor + batch_size > len(order):
                import random as _rnd

                _rnd.shuffle(order)
                cursor = 0
            idxs = order[cursor : cursor + batch_size]
            cursor += batch_size
            batch, gt_base, gt_warp = _gather(idxs)
            metrics = store.train_step(
                model, batch, opt, device, cfg, gt_base=gt_base, gt_warp=gt_warp
            )
            mean_kp = None
            if step % log_every == 0 or step == start_step + 1 or step == steps:
                mean_kp = _mean_kp()
                ratio = mean_kp / kp0 if kp0 > 0 else None
                is_ce = str(cfg.get("kp_loss") or "matching") == "paper_ce"
                lo, hi = ("ce_kp", "ce_dust") if is_ce else ("fn", "fp")
                terms = (
                    f"{lo}={metrics['loss_fn']:.6f} {hi}={metrics['loss_fp']:.6f} "
                    f"desc={metrics['loss_desc']:.6f}"
                )
                if metrics.get("n_mask"):
                    terms += (
                        f" cells={metrics['n_pos']}/{metrics['n_mask']}"
                    )
                if ratio is not None:
                    _safe_print(
                        f"overfit product step={step}/{steps} "
                        f"batch_kp={metrics['loss_kp']:.6f} mean_kp={mean_kp:.6f} "
                        f"ratio={ratio:.4f} {terms}",
                        flush=True,
                    )
                else:
                    _safe_print(
                        f"overfit product step={step}/{steps} "
                        f"batch_kp={metrics['loss_kp']:.6f} {terms}",
                        flush=True,
                    )
                if (
                    first_collapse_step is None
                    and kp0 > 0
                    and mean_kp <= 0.2 * kp0
                ):
                    first_collapse_step = step
                    _safe_print(
                        f"overfit product first_collapse_step={step} "
                        f"mean_kp={mean_kp:.6f} ratio={mean_kp/kp0:.4f}",
                        flush=True,
                    )
            row = {
                "step": step,
                **metrics,
                "mean_kp": mean_kp,
            }
            history.append(row)
            if log_path is not None:
                store.append_jsonl(log_path, row)
            just_collapsed = first_collapse_step == step
            do_eval = eval_every > 0 and (
                step % eval_every == 0 or step == steps or just_collapsed
            )
            if do_eval:
                evals.append(_overfit_eval(step))
            if ckpt_every > 0 and step % ckpt_every == 0:
                _save_latest()
            if first_collapse_step is not None and stop_on_collapse:
                step_n = {
                    "step": step,
                    "loss_kp": mean_kp if mean_kp is not None else metrics["loss_kp"],
                    "loss_total": metrics["loss_total"],
                    "loss_desc": metrics["loss_desc"],
                }
                break
            if step == steps:
                mean_kp = _mean_kp() if mean_kp is None else mean_kp
                step_n = {
                    "step": step,
                    "loss_kp": mean_kp,
                    "loss_total": metrics["loss_total"],
                    "loss_desc": metrics["loss_desc"],
                }

        if ckpt_every <= 0 or steps % ckpt_every != 0:
            _save_latest()
        kp_n = float(step_n["loss_kp"]) if step_n else None
        ratio = (kp_n / kp0) if kp0 and kp_n is not None and kp0 > 0 else None
        summary = {
            "ok": True,
            "ts": int(time.time()),
            "mode": "product",
            "collapsed": bool(first_collapse_step is not None),
            "first_collapse_step": first_collapse_step,
            "ratio_kp": ratio,
            "n_tiles": len(tiles),
            "n_angles": len(angles),
            "n_samples": len(samples),
            "batch_size": batch_size,
            "each_tile": False,
            "fresh_weights": "once",
            "steps": steps,
            "start_step": start_step,
            "steps_ran": len(history),
            "stop_on_collapse": stop_on_collapse,
            "ckpt_path": str(ckpt_path),
            "ckpt_every": ckpt_every,
            "tile_image": tile_image,
            "init_weights": str(args.init_weights) if args.init_weights else None,
            "angles": angles,
            "side": side,
            "tiles": [{"pair_id": int(t["pair_id"]), "loc": t.get("loc")} for t in tiles],
            "n_gt": n_gt,
            "desc_max_cells": int(cfg["desc_max_cells"]),
            "lr": float(cfg["lr"]),
            "step0": step0,
            "stepN": step_n,
            "eval_every": eval_every,
            "kp_loss": kp_loss,
            "evals": [
                {
                    "step": e.get("step"),
                    **{
                        kind: {
                            "n_pass": (e.get(kind) or {}).get("n_pass"),
                            "n_total": (e.get(kind) or {}).get("n_total"),
                            "pass_rate": (e.get(kind) or {}).get("pass_rate"),
                        }
                        for kind in ("nn", "lg")
                    },
                }
                for e in evals
            ],
            "log_path": str(log_path) if log_path else None,
            "eval_log_path": str(eval_log_path) if eval_log_path else None,
            "experiments_path": str(experiments_path),
        }
        store.append_jsonl(experiments_path, summary)
        emit(summary)
        return

    summaries = []
    tile_groups = [[t] for t in tiles] if bool(args.each_tile) else [tiles]
    for group in tile_groups:
        for theta in angles:
            _safe_print(
                f"=== overfit theta={theta:g} n_tiles={len(group)} ===",
                flush=True,
            )
            images = []
            warps = []
            masks = []
            Hs = []
            for tile in group:
                key = (int(tile["pair_id"]), side)
                base, warped, valid, H = data.make_warp_pair(
                    page_cache[key],
                    tile,
                    depth=int(cfg["depth"]),
                    preview_level=int(cfg["preview_level"]),
                    src_size=int(cfg["src_size"]),
                    out_size=int(cfg["out_size"]),
                    theta_deg=theta,
                )
                images.append(torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0))
                warps.append(torch.from_numpy(warped.astype(np.float32) / 255.0).unsqueeze(0))
                masks.append(torch.from_numpy((valid > 127).astype(np.float32)))
                Hs.append(torch.from_numpy(H))

            batch = {
                "image": torch.stack(images, dim=0),
                "warped": torch.stack(warps, dim=0),
                "valid_mask": torch.stack(masks, dim=0),
                "homography": torch.stack(Hs, dim=0),
                "theta_deg": [theta] * len(group),
                "pair_id": [int(t["pair_id"]) for t in group],
                "side": [side] * len(group),
            }

            model = _build_model(cfg, device)
            images_t = batch["image"].to(device)
            H_t = batch["homography"].to(device)
            with torch.no_grad():
                pseudo = store.detect_pseudo_gt(model, images_t, device)
                gt_base = store.filter_gt_in_frame(pseudo, int(cfg["out_size"]))
                gt_warp = store.filter_gt_in_frame(
                    store.warp_gt_points(pseudo, H_t), int(cfg["out_size"])
                )
            n_gt = int(sum(int(g.shape[0]) for g in gt_base))
            if n_gt == 0:
                raise RuntimeError(f"frozen pseudo-GT empty at theta={theta}")

            opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
            log_path = None
            if args.log:
                log_path = exp_dir / f"overfit_theta{theta:g}_n{len(group)}.jsonl"
                if log_path.is_file():
                    log_path.unlink()

            step0 = None
            step_n = None
            first_collapse_step = None
            history = []
            for step in range(1, steps + 1):
                metrics = store.train_step(
                    model, batch, opt, device, cfg, gt_base=gt_base, gt_warp=gt_warp
                )
                row = {"step": step, **metrics}
                history.append(row)
                if step == 1:
                    step0 = row
                if step == steps:
                    step_n = row
                if (
                    first_collapse_step is None
                    and step0 is not None
                    and float(step0["loss_kp"]) > 0
                    and float(metrics["loss_kp"]) <= 0.2 * float(step0["loss_kp"])
                ):
                    first_collapse_step = step
                    _safe_print(
                        f"overfit first_collapse_step={step} "
                        f"kp={metrics['loss_kp']:.6f} "
                        f"ratio={metrics['loss_kp']/step0['loss_kp']:.4f}",
                        flush=True,
                    )
                if step % log_every == 0 or step == 1 or step == steps:
                    _safe_print(
                        f"overfit theta={theta:g} n={len(group)} step={step}/{steps} "
                        f"total={metrics['loss_total']:.6f} "
                        f"kp={metrics['loss_kp']:.6f} desc={metrics['loss_desc']:.6f}",
                        flush=True,
                    )
                if log_path is not None:
                    store.append_jsonl(log_path, row)
                if first_collapse_step is not None:
                    step_n = row
                    break

            kp0 = float(step0["loss_kp"]) if step0 else None
            kp_n = float(step_n["loss_kp"]) if step_n else None
            ratio = (kp_n / kp0) if kp0 and kp0 > 0 else None
            summary = {
                "ok": True,
                "ts": int(time.time()),
                "mode": "single_batch",
                "collapsed": bool(first_collapse_step is not None),
                "first_collapse_step": first_collapse_step,
                "ratio_kp": ratio,
                "n_tiles": len(group),
                "each_tile": bool(args.each_tile),
                "fresh_weights": "once_per_run",
                "steps": steps,
                "steps_ran": len(history),
                "theta": theta,
                "side": side,
                "tiles": [
                    {"pair_id": int(t["pair_id"]), "loc": t.get("loc")} for t in group
                ],
                "n_gt": n_gt,
                "desc_max_cells": int(cfg["desc_max_cells"]),
                "lr": float(cfg["lr"]),
                "step0": step0,
                "stepN": step_n,
                "log_path": str(log_path) if log_path else None,
                "experiments_path": str(experiments_path),
            }
            store.append_jsonl(experiments_path, summary)
            summaries.append(summary)
            del model, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if len(summaries) == 1:
        emit(summaries[0])
    else:
        emit(
            {
                "ok": True,
                "n_runs": len(summaries),
                "experiments_path": str(experiments_path),
                "by_run": [
                    {
                        "theta": s["theta"],
                        "tiles": s["tiles"],
                        "first_collapse_step": s["first_collapse_step"],
                        "ratio_kp": s["ratio_kp"],
                        "collapsed": s["collapsed"],
                        "steps_ran": s["steps_ran"],
                    }
                    for s in summaries
                ],
            }
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--id", default=None)
    c.add_argument("--pairs", default=None)
    c.add_argument("--batch-size", type=int, default=None)
    c.add_argument("--lr", type=float, default=None)
    c.add_argument("--max-epochs", type=int, default=None)
    c.add_argument("--ckpt-every-epochs", type=int, default=None)
    c.add_argument("--eval-every-epochs", type=int, default=None)
    c.add_argument("--log-every", type=int, default=None)
    c.add_argument("--split-seed", type=int, default=None)
    c.add_argument("--eval-max-tiles", type=int, default=None)
    c.add_argument("--run-baseline", action="store_true")
    c.add_argument("--depth", type=int, default=None)
    c.add_argument("--max-steps", type=int, default=None)
    c.add_argument("--eval-every-steps", type=int, default=None)
    c.add_argument("--ckpt-every-steps", type=int, default=None)
    c.add_argument("--eval-step-tiles", type=int, default=None)
    c.add_argument("--eval-angles", default=None, help="comma list of degrees")
    c.add_argument("--desc-max-cells", type=int, default=None)
    c.add_argument("--num-workers", type=int, default=None)
    c.add_argument("--desc-lambda", type=float, default=None)
    c.add_argument("--kp-loss", default=None, choices=("matching", "paper_ce"))
    c.add_argument("--gt-nms-dist", type=int, default=None, help="teacher NMS radius for GT")
    c.add_argument("--gt-conf-thresh", type=float, default=None)
    c.add_argument(
        "--frozen-gt",
        action="store_true",
        help="take detector targets from a frozen teacher instead of the model being trained",
    )
    for name in ("run", "resume", "pause", "stop", "status"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
    lg = sub.add_parser("logs")
    lg.add_argument("run_id")
    lg.add_argument("-n", type=int, default=50)
    ov = sub.add_parser("overfit")
    ov.add_argument("--run-id", default="first")
    ov.add_argument("--theta", type=float, default=0.0)
    ov.add_argument(
        "--angles",
        default=None,
        help="comma-separated angles; overrides --theta when set",
    )
    ov.add_argument("--steps", type=int, default=200)
    ov.add_argument("--n-tiles", type=int, default=1)
    ov.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="minibatch size for product mode (tiles×angles, one shared model)",
    )
    ov.add_argument(
        "--each-tile",
        action="store_true",
        help="separate overfit per tile×angle with re-init (not shared weights)",
    )
    ov.add_argument("--lr", type=float, default=None)
    ov.add_argument("--side", default="he", choices=("he", "ihc"))
    ov.add_argument("--pair", type=int, default=None)
    ov.add_argument("--loc", default=None)
    ov.add_argument("--log-every", type=int, default=5)
    ov.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="product-mode rigid eval every N steps plus step 0; 0 disables",
    )
    ov.add_argument("--log", action="store_true")
    ov.add_argument(
        "--desc-max-cells",
        type=int,
        default=576,
        help="0 = dense Magicleap desc; default 576 subsample of source cells",
    )
    ov.add_argument(
        "--desc-lambda",
        type=float,
        default=None,
        help=(
            "positive weight of the desc hinge. The paper's 250 assumes a 30x40 grid; "
            "the pos/neg balance is lambda/(Hc*Wc), so scale it with resolution "
            "(64x64 needs ~853 to match)"
        ),
    )
    ov.add_argument(
        "--no-collapse-stop",
        action="store_true",
        help="record collapse but keep training to --steps",
    )
    ov.add_argument("--init-weights", default=None)
    ov.add_argument("--start-step", type=int, default=0)
    ov.add_argument("--tag", default="")
    ov.add_argument("--tile-image", default=None)
    ov.add_argument(
        "--src-size",
        type=int,
        default=None,
        help="pre-rotation crop; set equal to --out-size for same-FOV rotate-in-place",
    )
    ov.add_argument(
        "--out-size",
        type=int,
        default=None,
        help="final crop fed to SuperPoint (default from run config, usually 512)",
    )
    ov.add_argument(
        "--kp-loss",
        default="matching",
        choices=("matching", "paper_ce"),
        help="paper_ce = original SuperPoint 65-way -log softmax on encoded labels",
    )
    ov.add_argument(
        "--exp-subdir",
        default="_overfit",
        help="subdir under data/sp_rot_train for logs, weights and eval",
    )
    ov.add_argument(
        "--ckpt-every",
        type=int,
        default=0,
        help="overwrite one latest weight file every N steps; 0 = save only at end",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    {
        "list": cmd_list,
        "create": cmd_create,
        "run": cmd_run,
        "resume": cmd_resume,
        "pause": cmd_pause,
        "stop": cmd_stop,
        "status": cmd_status,
        "logs": cmd_logs,
        "overfit": cmd_overfit,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
