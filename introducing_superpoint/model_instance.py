"""
Training run metadata: config, per-epoch logs, checkpoint path, lineage.

Each ModelInstance owns one weights file at run_dir/name.pth, overwritten on
every save. training_log.json in run_dir holds the serialised epoch history.
"""
import json
import importlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import conf
importlib.reload(conf)

from superpoint_pytorch import default_config


DEFAULT_RUN_DIR = conf.resolve("introducing_superpoint/runs")
DEFAULT_WEIGHTS = conf.resolve("introducing_superpoint/superpoint_v6_from_tf.pth")


def checkpoint_timestamp(when=None) -> str:
    when = when or datetime.now()
    return when.strftime("%d-%m_%H-%M")


@dataclass
class TrainingConfig:
    name: str
    learning_rate: float = default_config["learning_rate"]
    batch_size: int = 8
    num_epochs: int = 5
    save_every_epochs: int = 1
    w_kp: float = 1.0
    w_loc: float = 1.0
    w_fn: float = 1.0
    w_fp: float = 5.0
    kp_radius: int = 8
    desc_patch_size: float = 3.0
    desc_centricity: float = 1.0
    num_workers: int = 0
    weights_init: Path = DEFAULT_WEIGHTS
    run_dir: Path = DEFAULT_RUN_DIR


@dataclass
class EpochLog:
    epoch: int
    loss_total: float
    loss_descriptor: float
    loss_keypoint: float
    loss_loc: float
    loss_fn: float
    loss_fp: float
    repeatability: float
    precision: float
    recall: float
    duration_seconds: float = 0.0


@dataclass
class ModelInstance:
    name: str
    config: TrainingConfig
    parent: Optional[str] = None
    epoch_logs: list[EpochLog] = field(default_factory=list)
    resume_epoch: int = 1
    last_pth_path: Optional[Path] = None

    @property
    def run_dir(self) -> Path:
        return self.config.run_dir / self.name

    @property
    def pth_path(self) -> Path:
        if self.last_pth_path is not None:
            return self.last_pth_path
        return self.run_dir / f"{self.name}.pth"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "training_log.json"

    def save_log(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "parent": self.parent,
            "config": _config_to_dict(self.config),
            "pth_path": conf.to_relative(self.pth_path),
            "resume_epoch": self.resume_epoch,
            "epoch_logs": [asdict(entry) for entry in self.epoch_logs],
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return self.log_path

    @classmethod
    def load_log(cls, log_path: Path) -> "ModelInstance":
        with open(log_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        config_dict = payload["config"]
        config = TrainingConfig(
            name=config_dict["name"],
            learning_rate=config_dict.get("learning_rate", default_config["learning_rate"]),
            batch_size=config_dict.get("batch_size", 4),
            num_epochs=config_dict.get("num_epochs", 1),
            save_every_epochs=config_dict.get("save_every_epochs", 1),
            w_kp=config_dict.get("w_kp", 1.0),
            w_loc=config_dict.get("w_loc", 1.0),
            w_fn=config_dict.get("w_fn", 1.0),
            w_fp=config_dict.get("w_fp", 5.0),
            kp_radius=config_dict.get("kp_radius", 8),
            desc_patch_size=config_dict.get("desc_patch_size", 3.0),
            desc_centricity=config_dict.get("desc_centricity", 1.0),
            num_workers=config_dict.get("num_workers", 0),
            weights_init=conf.resolve(config_dict.get("weights_init", conf.to_relative(DEFAULT_WEIGHTS))),
            run_dir=conf.resolve(config_dict.get("run_dir", conf.to_relative(DEFAULT_RUN_DIR))),
        )
        epoch_logs = []
        for entry in payload.get("epoch_logs", []):
            entry = dict(entry)
            entry.pop("gt_bin_recall", None)
            entry.setdefault("duration_seconds", 0.0)
            epoch_logs.append(EpochLog(**entry))
        resume_epoch = payload.get("resume_epoch")
        if resume_epoch is None:
            if len(epoch_logs) >= config.num_epochs:
                resume_epoch = config.num_epochs + 1
            else:
                resume_epoch = len(epoch_logs) + 1 if epoch_logs else 1
        last_pth_path = None
        if payload.get("pth_path"):
            last_pth_path = conf.resolve(payload["pth_path"])

        return cls(
            name=payload["name"],
            config=config,
            parent=payload.get("parent"),
            epoch_logs=epoch_logs,
            resume_epoch=resume_epoch,
            last_pth_path=last_pth_path,
        )

    def merge_saved_state(self) -> bool:
        if not self.log_path.exists():
            return False
        saved = self.load_log(self.log_path)
        self.epoch_logs = saved.epoch_logs
        self.resume_epoch = saved.resume_epoch
        self.last_pth_path = saved.last_pth_path
        if saved.parent is not None:
            self.parent = saved.parent
        return True

    def training_complete(self) -> bool:
        return len(self.epoch_logs) >= self.config.num_epochs


def should_resume_training(instance: ModelInstance) -> bool:
    if not instance.log_path.exists():
        return False
    saved = ModelInstance.load_log(instance.log_path)
    if len(saved.epoch_logs) >= instance.config.num_epochs:
        return False
    return saved.resume_epoch <= instance.config.num_epochs


def _config_to_dict(config: TrainingConfig) -> dict:
    data = asdict(config)
    data["weights_init"] = conf.to_relative(Path(config.weights_init))
    data["run_dir"] = conf.to_relative(Path(config.run_dir))
    return data
