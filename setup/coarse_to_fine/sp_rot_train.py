"""SP rot-inv fine-tune store + training step helpers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "introducing_superpoint"))

import conf
from setup.coarse_to_fine import sp_rot_bench as bench
from setup.coarse_to_fine import sp_rot_train_data as data

TRAIN_ROOT = conf.PROJECT_ROOT / "data" / "sp_rot_train"

DEFAULT_CONFIG = {
    "pairs": [0, 1, 3, 16],
    "depth": 5,
    "preview_level": 2,
    "src_size": 768,
    "out_size": 512,
    "extract_resize": 512,
    "sp_nms_dist": 8,
    "batch_size": 8,
    "lr": 0.0001,
    "max_epochs": 50,
    "ckpt_every_epochs": 1,
    "eval_every_epochs": 1,
    "log_every": 50,
    "split_seed": 0,
    "split_ratios": [0.8, 0.1, 0.1],
    "eval_max_tiles": 12,
    "eval_angles": [0, 90, 180, 270],
    "num_workers": 4,
    "desc_max_cells": 576,
    "w_kp": 1.0,
    "desc_lambda": 250.0,
    "desc_positive_margin": 1.0,
    "desc_negative_margin": 0.2,
    "kp_radius": 12,
    "w_loc": 1.0,
    "w_fn": 1.0,
    "w_fp": 0.5,
    "match_mode": "conf_distance",
    "match_epsilon": 1.0,
    "dataset": "muromi",
    "init_weights": str(conf.resolve("introducing_superpoint/superpoint_v6_from_tf.pth")),
    "skip_baseline": True,
}


def run_dir(run_id: str) -> Path:
    return TRAIN_ROOT / run_id


def config_path(run_id: str) -> Path:
    return run_dir(run_id) / "config.json"


def status_path(run_id: str) -> Path:
    return run_dir(run_id) / "status.json"


def split_path(run_id: str) -> Path:
    return run_dir(run_id) / "split.json"


def loss_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "loss.jsonl"


def eval_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "eval.jsonl"


def epoch_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "epoch.jsonl"


def ckpt_dir(run_id: str) -> Path:
    return run_dir(run_id) / "checkpoints"


def pause_flag(run_id: str) -> Path:
    return run_dir(run_id) / "PAUSE"


def stop_flag(run_id: str) -> Path:
    return run_dir(run_id) / "STOP"


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bench.dumps(obj, indent=2))


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(bench.json_safe(obj), separators=(",", ":")) + "\n")


def _migrate_config(cfg: dict) -> dict:
    aliases = {
        "b1_every_epochs": "eval_every_epochs",
        "b1_max_tiles": "eval_max_tiles",
        "b1_angles": "eval_angles",
    }
    out = dict(cfg)
    for old, new in aliases.items():
        if old in out and new not in out:
            out[new] = out[old]
        out.pop(old, None)
    return out


def create_run(name: str, config: dict | None = None, run_id: str | None = None) -> dict:
    cfg = {**DEFAULT_CONFIG, **_migrate_config(config or {})}
    rid = bench.slugify(run_id or name)
    root = run_dir(rid)
    if root.exists():
        raise FileExistsError(rid)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    ckpt_dir(rid).mkdir()

    tiles = data.scan_tiles(list(cfg["pairs"]), int(cfg["depth"]))
    if not tiles:
        raise RuntimeError(f"no unmasked tiles for pairs={cfg['pairs']} depth={cfg['depth']}")
    ratios = tuple(float(x) for x in cfg.get("split_ratios") or DEFAULT_CONFIG["split_ratios"])
    split = data.split_tiles(tiles, ratios=ratios, seed=int(cfg.get("split_seed") or 0))
    write_json(split_path(rid), split)

    man = {
        "id": rid,
        "name": name.strip() or rid,
        "created_at": int(time.time()),
        **cfg,
        "n_total": split["n_total"],
        "n_train": split["n_train"],
        "n_val": split["n_val"],
        "n_test": split["n_test"],
    }
    write_json(config_path(rid), man)
    write_json(
        status_path(rid),
        {
            "state": "created",
            "step": 0,
            "epoch": 0,
            "detail": None,
            "error": None,
            "last_eval": None,
            "last_epoch_s": None,
            "updated_at": int(time.time()),
        },
    )
    return man


def load_config(run_id: str) -> dict:
    cfg = read_json(config_path(run_id))
    if not cfg:
        raise FileNotFoundError(run_id)
    return {**DEFAULT_CONFIG, **_migrate_config(cfg)}


def load_split(run_id: str) -> dict:
    sp = read_json(split_path(run_id))
    if not sp:
        raise FileNotFoundError(f"split missing for {run_id}")
    return sp


def update_status(run_id: str, **kwargs) -> dict:
    st = read_json(status_path(run_id)) or {}
    st.update(kwargs)
    st["updated_at"] = int(time.time())
    write_json(status_path(run_id), st)
    return st


def list_runs() -> list[dict]:
    if not TRAIN_ROOT.is_dir():
        return []
    out = []
    for p in sorted(TRAIN_ROOT.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        cfg = read_json(p / "config.json")
        if cfg:
            cfg["status"] = read_json(p / "status.json")
            out.append(cfg)
    return out


def save_checkpoint(run_id: str, model: torch.nn.Module, step: int, epoch: int = 0) -> Path:
    d = ckpt_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step_{step}.pt"
    latest = d / "latest.pt"
    sd = model.state_dict()
    torch.save(sd, path)
    torch.save(sd, latest)
    meta = {"step": step, "epoch": epoch, "path": str(path), "saved_at": int(time.time())}
    write_json(d / "latest.json", meta)
    return path


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    sd = torch.load(str(path), map_location=device)
    model.load_state_dict(sd, strict=False)


def detect_pseudo_gt(model, images: torch.Tensor, device: torch.device, max_kpts: int = 512):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model({"image": images.to(device)}, training=False)
    model.train(was_training)
    gts = []
    for kpts, scores in zip(out["keypoints"], out["keypoint_scores"]):
        if kpts.numel() == 0:
            gts.append(torch.zeros((0, 3), device=device))
            continue
        if kpts.shape[0] > max_kpts:
            scores, idx = torch.topk(scores, max_kpts)
            kpts = kpts[idx]
        conf = scores.clamp(0.05, 1.0)
        gts.append(torch.cat([kpts, conf.unsqueeze(1)], dim=1))
    return gts


def warp_gt_points(gts: list[torch.Tensor], homographies: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for b, gt in enumerate(gts):
        if gt.numel() == 0:
            out.append(gt)
            continue
        H = homographies[b].to(gt.device, dtype=gt.dtype)
        ones = torch.ones(gt.shape[0], 1, device=gt.device, dtype=gt.dtype)
        pts = torch.cat([gt[:, :2], ones], dim=1)
        mapped = pts @ H.T
        xy = mapped[:, :2] / mapped[:, 2:3].clamp(min=1e-6)
        out.append(torch.cat([xy, gt[:, 2:3]], dim=1))
    return out


def filter_gt_in_frame(gts: list[torch.Tensor], size: int) -> list[torch.Tensor]:
    out = []
    for gt in gts:
        if gt.numel() == 0:
            out.append(gt)
            continue
        m = (
            (gt[:, 0] >= 0)
            & (gt[:, 0] < size)
            & (gt[:, 1] >= 0)
            & (gt[:, 1] < size)
        )
        out.append(gt[m])
    return out


def _forward_losses(
    model,
    batch,
    device,
    cfg: dict,
    *,
    gt_base=None,
    gt_warp=None,
) -> dict:
    import utils
    from superpoint_pytorch import default_config

    images = batch["image"].to(device)
    warped = batch["warped"].to(device)
    valid = batch["valid_mask"].to(device)
    H = batch["homography"].to(device)

    if gt_base is None or gt_warp is None:
        pseudo = detect_pseudo_gt(model, images, device)
        gt_base = filter_gt_in_frame(pseudo, int(cfg["out_size"]))
        gt_warp = filter_gt_in_frame(warp_gt_points(pseudo, H), int(cfg["out_size"]))

    out0 = model({"image": images}, training=True)
    out1 = model({"image": warped}, training=True)

    kp_kwargs = {
        "cell_size": default_config["grid_size"],
        "radius": int(cfg["kp_radius"]),
        "w_loc": float(cfg["w_loc"]),
        "w_fn": float(cfg["w_fn"]),
        "w_fp": float(cfg["w_fp"]),
        "match_mode": cfg["match_mode"],
        "match_epsilon": float(cfg["match_epsilon"]),
    }
    kp0 = utils.keypoint_matching_loss_detailed(out0["logits"], gt_base, **kp_kwargs)
    kp1 = utils.keypoint_matching_loss_detailed(out1["logits"], gt_warp, **kp_kwargs)
    desc_max = cfg.get("desc_max_cells")
    desc_cfg = {
        **default_config,
        "lambda_d": float(cfg["desc_lambda"]),
        "positive_margin": float(cfg["desc_positive_margin"]),
        "negative_margin": float(cfg["desc_negative_margin"]),
        "desc_max_cells": int(desc_max) if desc_max is not None else 576,
    }
    desc = utils.descriptor_loss(
        out0["descriptors_raw"],
        out1["descriptors_raw"],
        desc_cfg,
        valid_mask=valid,
        homographies=H,
    )
    loss_kp = kp0["loss"] + kp1["loss"]
    loss = float(cfg["w_kp"]) * loss_kp + desc
    return {
        "loss": loss,
        "loss_kp": loss_kp,
        "loss_desc": desc,
        "theta_mean": float(sum(batch["theta_deg"]) / max(1, len(batch["theta_deg"]))),
    }


def train_step(
    model,
    batch,
    optimizer,
    device,
    cfg: dict,
    *,
    gt_base=None,
    gt_warp=None,
) -> dict:
    model.train()
    parts = _forward_losses(
        model, batch, device, cfg, gt_base=gt_base, gt_warp=gt_warp
    )
    loss = parts["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "loss_total": float(loss.detach().cpu()),
        "loss_kp": float(parts["loss_kp"].detach().cpu()),
        "loss_desc": float(parts["loss_desc"].detach().cpu()),
        "theta_mean": parts["theta_mean"],
    }


@torch.no_grad()
def eval_loss(model, loader, device, cfg: dict) -> dict:
    model.eval()
    totals = {"loss_total": 0.0, "loss_kp": 0.0, "loss_desc": 0.0}
    n = 0
    for batch in loader:
        parts = _forward_losses(model, batch, device, cfg)
        totals["loss_total"] += float(parts["loss"].detach().cpu())
        totals["loss_kp"] += float(parts["loss_kp"].detach().cpu())
        totals["loss_desc"] += float(parts["loss_desc"].detach().cpu())
        n += 1
    if n == 0:
        return {"loss_total": None, "loss_kp": None, "loss_desc": None, "n_batches": 0}
    return {k: v / n for k, v in totals.items()} | {"n_batches": n}
