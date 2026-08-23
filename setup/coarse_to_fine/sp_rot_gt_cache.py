"""
Frozen pseudo-GT for the rotation fine-tune, computed lazily and persisted.

Detector targets must come from the *original* weights, never from the model being
trained, or the labels drift with the weights. A frozen teacher supplies them.

Targets are stored as NMS keypoint coordinates at 0 rotation, keyed by
(pair, side, loc): a 65-way cell encoding cannot be rotated, so the warp and
`encode_keypoint_labels` still run per sampled angle at training time.

The cache fills on first sight of each crop and is written to disk per pair, so a
restart pays nothing and no separate prebuild step is needed. The teacher forward
is amortised: ~8k unique crops against 160k sample draws in a 20k-step run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "introducing_superpoint"))

import conf

CACHE_ROOT = conf.PROJECT_ROOT / "data" / "sp_rot_train" / "_gt_cache"


class GtCache:
    """Frozen-teacher keypoint cache. `get(batch)` returns one (N,3) tensor per sample."""

    def __init__(self, cfg: dict, device: torch.device, root: Path | None = None):
        from setup.coarse_to_fine import sp_rot_train as store
        from training import build_model

        self._store = store
        self.cfg = cfg
        self.device = device
        self.out_size = int(cfg["out_size"])
        self.dir = Path(root) if root is not None else CACHE_ROOT / self.tag(cfg)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.teacher = build_model(
            cfg["init_weights"],
            device=device,
            nms_radius=int(cfg["gt_nms_dist"]),
            detection_threshold=float(cfg["gt_conf_thresh"]),
            max_num_keypoints=cfg.get("gt_max_kpts"),
        ).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.mem: dict[tuple[int, str, str], torch.Tensor] = {}
        self._loaded: set[int] = set()
        self._dirty: set[int] = set()
        self.n_hit = 0
        self.n_miss = 0

    @staticmethod
    def tag(cfg: dict) -> str:
        """Teacher settings live in the path so changed GT cannot silently reuse stale entries."""
        return (
            f"d{int(cfg['depth'])}_l{int(cfg['preview_level'])}_s{int(cfg['src_size'])}"
            f"_o{int(cfg['out_size'])}_nms{int(cfg['gt_nms_dist'])}"
            f"_th{float(cfg['gt_conf_thresh']):g}_k{cfg.get('gt_max_kpts')}"
        )

    def _pair_path(self, pair_id: int) -> Path:
        return self.dir / f"pair{int(pair_id)}.pt"

    def _ensure_loaded(self, pair_id: int) -> None:
        if pair_id in self._loaded:
            return
        self._loaded.add(pair_id)
        path = self._pair_path(pair_id)
        if not path.is_file():
            return
        try:
            entries = torch.load(str(path), map_location="cpu").get("entries") or {}
        except Exception:
            return
        for k, v in entries.items():
            side, loc = k.split("/", 1)
            self.mem.setdefault((pair_id, side, loc), v)

    @torch.no_grad()
    def get(self, batch: dict) -> list[torch.Tensor]:
        pids = [int(p) for p in batch["pair_id"]]
        sides = list(batch["side"])
        locs = [str(l) for l in batch["loc"]]
        for pid in set(pids):
            self._ensure_loaded(pid)

        keys = list(zip(pids, sides, locs))
        miss = [i for i, k in enumerate(keys) if k not in self.mem]
        self.n_hit += len(keys) - len(miss)
        self.n_miss += len(miss)
        if miss:
            imgs = batch["image"][miss].to(self.device)
            gt = self._store.detect_pseudo_gt(self.teacher, imgs, self.device)
            gt = self._store.filter_gt_in_frame(gt, self.out_size)
            for j, i in enumerate(miss):
                self.mem[keys[i]] = gt[j].detach().cpu()
                self._dirty.add(keys[i][0])
        return [self.mem[k].to(self.device) for k in keys]

    def flush(self) -> int:
        n = 0
        for pid in sorted(self._dirty):
            entries = {
                f"{side}/{loc}": v for (p, side, loc), v in self.mem.items() if p == pid
            }
            torch.save(
                {
                    "pair_id": pid,
                    "tag": self.tag(self.cfg),
                    "init_weights": str(self.cfg["init_weights"]),
                    "entries": entries,
                },
                str(self._pair_path(pid)),
            )
            n += len(entries)
        self._dirty.clear()
        return n

    def stats(self) -> dict:
        counts = [int(v.shape[0]) for v in self.mem.values()]
        return {
            "n_entries": len(self.mem),
            "n_hit": self.n_hit,
            "n_miss": self.n_miss,
            "kp_mean": (sum(counts) / len(counts)) if counts else 0.0,
            "n_empty": sum(1 for c in counts if c == 0),
        }
