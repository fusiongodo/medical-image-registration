"""
Launch SuperPoint training with env-configurable TrainingConfig.

RUN_NAME=runpod NUM_EPOCHS=20 BATCH_SIZE=4 NUM_WORKERS=4 DATA_PREFIX=macos \\
  python remote/launch_training.py
"""
import os
import sys
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "introducing_superpoint"))

import conf
importlib.reload(conf)

from model_instance import ModelInstance, TrainingConfig
import training


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def main():
    run_name = os.environ.get("RUN_NAME", "runpod")
    config = TrainingConfig(
        name=run_name,
        num_epochs=_env_int("NUM_EPOCHS", 20),
        batch_size=_env_int("BATCH_SIZE", 4),
        num_workers=_env_int("NUM_WORKERS", 4),
        kpi_every_instances=_env_int("KPI_EVERY", 720),
        learning_rate=_env_float("LR", 0.001),
        save_every_epochs=_env_int("SAVE_EVERY_EPOCHS", 1),
        w_kp=_env_float("W_KP", 1.0),
    )
    parent = os.environ.get("PARENT_RUN", "superpoint_v6_from_tf")
    resume = os.environ.get("RESUME", "auto")
    if resume == "auto":
        resume = None
    elif resume in ("0", "false", "no"):
        resume = False
    else:
        resume = True

    instance = ModelInstance(name=run_name, config=config, parent=parent)

    try:
        instance, model = training.train_model(instance, resume=resume)
    except KeyboardInterrupt:
        raise SystemExit(130)

    print(f"checkpoint : {instance.pth_path}")
    print(f"log        : {instance.log_path}")
    return instance, model


if __name__ == "__main__":
    main()
