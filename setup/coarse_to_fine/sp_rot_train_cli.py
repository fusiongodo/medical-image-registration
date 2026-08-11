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
    from training import build_model

    return build_model(cfg.get("init_weights"), device=device)


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
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    max_epochs = int(cfg["max_epochs"])
    store.update_status(
        run_id, state="running", step=step, epoch=completed_epochs, detail="start", error=None
    )

    if completed_epochs == 0 and step == 0 and not bool(cfg.get("skip_baseline")):
        _run_eval(run_id, model, device, cfg, val_tiles, kind="baseline", step=0, epoch=0)

    while completed_epochs < max_epochs:
        cfg = store.load_config(run_id)
        max_epochs = int(cfg["max_epochs"])
        for g in opt.param_groups:
            g["lr"] = float(cfg["lr"])

        epoch = completed_epochs + 1
        t0 = time.time()
        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            if store.stop_flag(run_id).is_file():
                store.save_checkpoint(run_id, model, step, completed_epochs)
                store.update_status(
                    run_id, state="stopped", step=step, epoch=completed_epochs, detail="stopped"
                )
                emit({"ok": True, "state": "stopped", "step": step, "epoch": completed_epochs})
                return
            if store.pause_flag(run_id).is_file():
                store.save_checkpoint(run_id, model, step, completed_epochs)
                store.pause_flag(run_id).unlink(missing_ok=True)
                store.update_status(
                    run_id, state="paused", step=step, epoch=completed_epochs, detail="paused"
                )
                emit({"ok": True, "state": "paused", "step": step, "epoch": completed_epochs})
                return

            cfg = store.load_config(run_id)
            for g in opt.param_groups:
                g["lr"] = float(cfg["lr"])
            metrics = store.train_step(model, batch, opt, device, cfg)
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

        epoch_s = time.time() - t0
        val_m = store.eval_loss(model, val_loader, device, cfg)
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
    for name in ("run", "resume", "pause", "stop", "status"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
    lg = sub.add_parser("logs")
    lg.add_argument("run_id")
    lg.add_argument("-n", type=int, default=50)
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
    }[args.cmd](args)


if __name__ == "__main__":
    main()
