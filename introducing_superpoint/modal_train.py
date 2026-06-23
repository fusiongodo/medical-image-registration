"""
Modal training runner for SuperPoint stain-invariant fine-tuning.

Volumes (create once):
    modal volume create sp-data   # data lives here
    modal volume create sp-runs   # checkpoints + logs written here

Upload data as a single zip (fast — one HTTP request):
    zip -r cropped_smooth.zip data/cropped_smooth/
    modal volume put sp-data data/cropped_smooth_training.zip /cropped_smooth_training.zip

Unzip on Modal (one-time, free CPU container):
    modal run introducing_superpoint/modal_train.py --unzip

Run training:
    modal run introducing_superpoint/modal_train.py
    modal run introducing_superpoint/modal_train.py --name run1 --epochs 50 --batch 64

Fetch results:
    modal volume get sp-runs /smoke ./runs/smoke
"""
import sys
from pathlib import Path

import modal

# ── Volumes ──────────────────────────────────────────────────────────────────

data_vol = modal.Volume.from_name("sp-data", create_if_missing=True)
runs_vol  = modal.Volume.from_name("sp-runs",  create_if_missing=True)

DATA_MOUNT = "/sp-data"
RUNS_MOUNT = "/sp-runs"

# ── Image ─────────────────────────────────────────────────────────────────────

_repo_root = Path(__file__).parent.parent

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime")
    .pip_install(
        "Pillow",
        "numpy",
        "opencv-python-headless",
        "torchvision",
        "scipy",
        "matplotlib",
    )
    .add_local_dir(
        _repo_root,
        remote_path="/repo",
        ignore=["data", ".venv", ".git", "introducing_superpoint/runs",
                "quadtree-level-validation/node_modules"],
    )
)

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("superpoint-training", image=image)

# ── Training function ─────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    cpu=8.0,
    timeout=60 * 60 * 12,
    volumes={
        DATA_MOUNT: data_vol,
        RUNS_MOUNT: runs_vol,
    },
)
def train(
    name:       str   = "smoke",
    epochs:     int   = 20,
    batch:      int   = 32,
    lr:         float = 0.003,
    workers:    int   = 0,
    match_mode: str   = "conf_distance",
):
    sys.path.insert(0, "/repo")
    sys.path.insert(0, "/repo/introducing_superpoint")

    import conf
    from pathlib import Path as P

    conf.PROJECT_ROOT = P("/repo")

    from dataset import StainPairKeypointDataset
    from model_instance import ModelInstance, TrainingConfig
    from training import train_model

    cropped_smooth = P(DATA_MOUNT) / "cropped_smooth_training"

    run_dir = P(RUNS_MOUNT) / name

    config = TrainingConfig(
        name=name,
        learning_rate=lr,
        batch_size=batch,
        num_epochs=epochs,
        num_workers=workers,
        match_mode=match_mode,
        run_dir=P(RUNS_MOUNT),
    )
    instance = ModelInstance(name=name, config=config, parent="superpoint_v6_from_tf")

    train_dataset = StainPairKeypointDataset(cropped_dir=cropped_smooth, split="train", preload=True)
    val_dataset   = StainPairKeypointDataset(cropped_dir=cropped_smooth, split="val",   preload=True)

    print(f"train tiles: {len(train_dataset)}  val tiles: {len(val_dataset)}")

    train_model(instance, train_dataset=train_dataset, val_dataset=val_dataset)

    runs_vol.commit()
    print(f"done — results in volume sp-runs at /{name}/")


# ── Unzip helper ─────────────────────────────────────────────────────────────

@app.function(
    volumes={DATA_MOUNT: data_vol},
    timeout=60 * 30,
)
def unzip_data():
    import zipfile
    from pathlib import Path as P

    zip_path  = P(DATA_MOUNT) / "cropped_smooth_training.zip"
    dest      = P(DATA_MOUNT)

    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found in volume — upload it first")

    print(f"extracting {zip_path} → {dest} …", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"{len(members)} files to extract", flush=True)
        zf.extractall(dest)

    data_vol.commit()
    print("done", flush=True)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    name:       str   = "smoke",
    epochs:     int   = 20,
    batch:      int   = 32,
    lr:         float = 0.003,
    workers:    int   = 0,
    match_mode: str   = "conf_distance",
    unzip:      bool  = False,
):
    if unzip:
        unzip_data.remote()
        return

    train.remote(
        name=name,
        epochs=epochs,
        batch=batch,
        lr=lr,
        workers=workers,
        match_mode=match_mode,
    )
