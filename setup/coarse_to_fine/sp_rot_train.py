"""SP rot-inv fine-tune store + training step helpers."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "introducing_superpoint"))

import conf
from setup.coarse_to_fine import sp_rot_bench as bench

TRAIN_ROOT = conf.PROJECT_ROOT / "data" / "sp_rot_train"

DEFAULT_CONFIG = {
    "pairs": [0, 1, 3, 16],
    "depth": 5,
    "preview_level": 2,
    "src_size": 768,
    "out_size": 512,
    "extract_resize": 512,
    "sp_nms_dist": 8,
    "batch_size": 4,
    "lr": 0.0001,
    "max_steps": 100000,
    "ckpt_every": 5000,
    "smoke_every": 5000,
    "full_every": 20000,
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
    "skip_baseline": False,
}


def run_dir(run_id: str) -> Path:
    return TRAIN_ROOT / run_id


def config_path(run_id: str) -> Path:
    return run_dir(run_id) / "config.json"


def status_path(run_id: str) -> Path:
    return run_dir(run_id) / "status.json"


def loss_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "loss.jsonl"


def eval_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "eval.jsonl"


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


def create_run(name: str, config: dict | None = None, run_id: str | None = None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rid = bench.slugify(run_id or name)
    root = run_dir(rid)
    if root.exists():
        raise FileExistsError(rid)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    ckpt_dir(rid).mkdir()
    man = {
        "id": rid,
        "name": name.strip() or rid,
        "created_at": int(time.time()),
        **cfg,
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
            "updated_at": int(time.time()),
        },
    )
    return man


def load_config(run_id: str) -> dict:
    cfg = read_json(config_path(run_id))
    if not cfg:
        raise FileNotFoundError(run_id)
    return {**DEFAULT_CONFIG, **cfg}


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


def save_checkpoint(run_id: str, model: torch.nn.Module, step: int) -> Path:
    d = ckpt_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step_{step}.pt"
    latest = d / "latest.pt"
    sd = model.state_dict()
    torch.save(sd, path)
    torch.save(sd, latest)
    meta = {"step": step, "path": str(path), "saved_at": int(time.time())}
    write_json(d / "latest.json", meta)
    return path


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    sd = torch.load(str(path), map_location=device)
    model.load_state_dict(sd, strict=False)


def detect_pseudo_gt(model, images: torch.Tensor, device: torch.device, max_kpts: int = 512):
    """Detached inference keypoints as (x,y,score) lists for KP loss."""
    model.eval()
    with torch.no_grad():
        out = model({"image": images.to(device)}, training=False)
    model.train()
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


def train_step(model, batch, optimizer, device, cfg: dict) -> dict:
    import utils
    from superpoint_pytorch import default_config

    images = batch["image"].to(device)
    warped = batch["warped"].to(device)
    valid = batch["valid_mask"].to(device)
    H = batch["homography"].to(device)

    pseudo = detect_pseudo_gt(model, images, device)
    gt_base = filter_gt_in_frame(pseudo, int(cfg["out_size"]))
    gt_warp = filter_gt_in_frame(warp_gt_points(pseudo, H), int(cfg["out_size"]))

    model.train()
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
    desc_cfg = {
        **default_config,
        "lambda_d": float(cfg["desc_lambda"]),
        "positive_margin": float(cfg["desc_positive_margin"]),
        "negative_margin": float(cfg["desc_negative_margin"]),
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
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "loss_total": float(loss.detach().cpu()),
        "loss_kp": float(loss_kp.detach().cpu()),
        "loss_desc": float(desc.detach().cpu()),
        "theta_mean": float(sum(batch["theta_deg"]) / max(1, len(batch["theta_deg"]))),
    }
