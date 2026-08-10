"""
SP rotational-invariance training CLI.

  python setup/coarse_to_fine/sp_rot_train_cli.py create --name smoke
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


def emit(obj) -> None:
    print(bench.dumps(obj, indent=2), flush=True)


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
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.ckpt_every is not None:
        cfg["ckpt_every"] = int(args.ckpt_every)
    if args.smoke_every is not None:
        cfg["smoke_every"] = int(args.smoke_every)
    if args.full_every is not None:
        cfg["full_every"] = int(args.full_every)
    if args.skip_baseline:
        cfg["skip_baseline"] = True
    try:
        man = store.create_run(args.name, config=cfg, run_id=args.id)
        emit({"ok": True, "run": man})
    except FileExistsError as e:
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
        }
    )


def _tail_jsonl(path: Path, n: int = 50) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    out = []
    for line in lines[-n:]:
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
        }
    )


def _build_model(cfg: dict, device: torch.device):
    from training import build_model

    return build_model(cfg.get("init_weights"), device=device)


def _run_loop(run_id: str, *, resume: bool) -> None:
    cfg = store.load_config(run_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, device)
    step = 0
    st = store.read_json(store.status_path(run_id)) or {}
    latest = store.ckpt_dir(run_id) / "latest.pt"
    if resume and latest.is_file():
        store.load_checkpoint(model, latest, device)
        step = int(st.get("step") or 0)
        meta = store.read_json(store.ckpt_dir(run_id) / "latest.json") or {}
        if meta.get("step") is not None:
            step = int(meta["step"])
    elif latest.is_file() is False and Path(cfg["init_weights"]).is_file():
        pass

    for flag in (store.pause_flag(run_id), store.stop_flag(run_id)):
        if flag.is_file():
            flag.unlink()

    ds = data.RotWarpDataset(
        pairs=list(cfg["pairs"]),
        depth=int(cfg["depth"]),
        preview_level=int(cfg["preview_level"]),
        src_size=int(cfg["src_size"]),
        out_size=int(cfg["out_size"]),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=data.collate_rot,
        drop_last=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    max_steps = int(cfg["max_steps"])
    store.update_status(run_id, state="running", step=step, detail="start", error=None)

    if step == 0 and not bool(cfg.get("skip_baseline")):
        print("baseline Magicleap full B1…", flush=True)
        base = beval.full_eval(
            cfg["init_weights"],
            pairs=list(cfg["pairs"]),
            extract_resize=int(cfg["extract_resize"]),
            nms=int(cfg["sp_nms_dist"]),
        )
        base["kind"] = "baseline_magicleap"
        base["step"] = 0
        store.append_jsonl(store.eval_log_path(run_id), base)
        store.update_status(run_id, last_eval=base)

    epoch = int(st.get("epoch") or 0)
    while step < max_steps:
        epoch += 1
        for batch in loader:
            if store.stop_flag(run_id).is_file():
                store.save_checkpoint(run_id, model, step)
                store.update_status(run_id, state="stopped", step=step, epoch=epoch, detail="stopped")
                emit({"ok": True, "state": "stopped", "step": step})
                return
            if store.pause_flag(run_id).is_file():
                store.save_checkpoint(run_id, model, step)
                store.pause_flag(run_id).unlink(missing_ok=True)
                store.update_status(run_id, state="paused", step=step, epoch=epoch, detail="paused")
                emit({"ok": True, "state": "paused", "step": step})
                return

            cfg = store.load_config(run_id)
            metrics = store.train_step(model, batch, opt, device, cfg)
            step += 1
            metrics.update(
                {
                    "step": step,
                    "epoch": epoch,
                    "lr": float(cfg["lr"]),
                    "ts": int(time.time()),
                }
            )
            store.append_jsonl(store.loss_log_path(run_id), metrics)
            store.update_status(
                run_id,
                state="running",
                step=step,
                epoch=epoch,
                detail=f"loss={metrics['loss_total']:.4f}",
            )

            if step % int(cfg["ckpt_every"]) == 0:
                store.save_checkpoint(run_id, model, step)
                print(f"ckpt step={step}", flush=True)

            if step % int(cfg["smoke_every"]) == 0:
                ckpt = store.ckpt_dir(run_id) / "latest.pt"
                store.save_checkpoint(run_id, model, step)
                print(f"smoke B1 step={step}", flush=True)
                ev = beval.smoke_eval(
                    ckpt,
                    pairs=list(cfg["pairs"])[:2],
                    extract_resize=int(cfg["extract_resize"]),
                    nms=int(cfg["sp_nms_dist"]),
                )
                ev["kind"] = "smoke"
                ev["step"] = step
                store.append_jsonl(store.eval_log_path(run_id), ev)
                store.update_status(run_id, last_eval=ev)

            if step % int(cfg["full_every"]) == 0:
                ckpt = store.ckpt_dir(run_id) / "latest.pt"
                store.save_checkpoint(run_id, model, step)
                print(f"full B1 step={step}", flush=True)
                ev = beval.full_eval(
                    ckpt,
                    pairs=list(cfg["pairs"]),
                    extract_resize=int(cfg["extract_resize"]),
                    nms=int(cfg["sp_nms_dist"]),
                )
                ev["kind"] = "full"
                ev["step"] = step
                store.append_jsonl(store.eval_log_path(run_id), ev)
                store.update_status(run_id, last_eval=ev)

            if step >= max_steps:
                break

    store.save_checkpoint(run_id, model, step)
    ev = None
    if not bool(cfg.get("skip_baseline")):
        print("final full B1…", flush=True)
        ev = beval.full_eval(
            store.ckpt_dir(run_id) / "latest.pt",
            pairs=list(cfg["pairs"]),
            extract_resize=int(cfg["extract_resize"]),
            nms=int(cfg["sp_nms_dist"]),
        )
        ev["kind"] = "final"
        ev["step"] = step
        store.append_jsonl(store.eval_log_path(run_id), ev)
    store.update_status(
        run_id,
        state="done",
        step=step,
        epoch=epoch,
        last_eval=ev,
        detail="finished",
    )
    emit({"ok": True, "state": "done", "step": step, "last_eval": ev})


def cmd_run(args: argparse.Namespace) -> None:
    try:
        _run_loop(args.run_id, resume=False)
    except Exception as e:
        store.update_status(args.run_id, state="error", error=str(e), detail="error")
        emit({"ok": False, "error": str(e)})
        raise


def cmd_resume(args: argparse.Namespace) -> None:
    try:
        _run_loop(args.run_id, resume=True)
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
    c.add_argument("--max-steps", type=int, default=None)
    c.add_argument("--ckpt-every", type=int, default=None)
    c.add_argument("--smoke-every", type=int, default=None)
    c.add_argument("--full-every", type=int, default=None)
    c.add_argument("--skip-baseline", action="store_true")
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
