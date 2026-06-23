"""
SuperPoint stain-invariant training: model load, loss assembly, epoch loop,
checkpointing, and KPI logging via ModelInstance.
"""
import sys
import time
import importlib
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
import conf
importlib.reload(conf)

from superpoint_pytorch import SuperPoint, default_config
from model_instance import (
    EpochLog,
    ModelInstance,
    TrainingConfig,
    checkpoint_timestamp,
    load_existing_run,
    latest_checkpoint_path,
    next_epoch_number,
)
from torch.utils.data import DataLoader
from dataset import StainPairKeypointDataset, collate_pairs, make_loader
import utils

DEFAULT_WEIGHTS = conf.resolve("introducing_superpoint/superpoint_v6_from_tf.pth")
PROGRESS_EVERY_BATCHES = 10


def build_model(weights_path=DEFAULT_WEIGHTS, device="cpu"):
    model = SuperPoint()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)



def _match_kwargs(config):
    return {
        "cell_size": default_config["grid_size"],
        "radius":    config.kp_radius,
        "match_mode": config.match_mode,
        "epsilon":   config.match_epsilon,
    }


def _desc_config(superpoint_config, training_config):
    return {
        **superpoint_config,
        "desc_patch_size": training_config.desc_patch_size,
        "desc_centricity": training_config.desc_centricity,
    }


def _fresh_kpi_totals():
    return {
        "repeatable": 0,
        "total_gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def _fresh_depth_totals():
    return {}


def _ensure_depth(totals_by_depth, depth_key):
    if depth_key not in totals_by_depth:
        totals_by_depth[depth_key] = _fresh_kpi_totals()


def _accumulate_batch_kpis(totals, logits_he, logits_ihc, gt_list, kp_kwargs):
    for b in range(logits_he.shape[0]):
        match_he = utils._match_keypoints_single(logits_he[b], gt_list[b], **kp_kwargs)
        match_ihc = utils._match_keypoints_single(logits_ihc[b], gt_list[b], **kp_kwargs)
        totals["repeatable"] += len(match_he["matched_gt_ids"] & match_ihc["matched_gt_ids"])
        totals["total_gt"] += match_he["num_gt"]
        totals["tp"] += match_he["tp"] + match_ihc["tp"]
        totals["fp"] += match_he["fp"] + match_ihc["fp"]
        totals["fn"] += match_he["fn"] + match_ihc["fn"]
    return totals


def _kpis_from_totals(totals):
    tp = totals["tp"]
    fp = totals["fp"]
    fn = totals["fn"]
    return {
        "repeatability": totals["repeatable"] / (totals["total_gt"] + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }


def total_loss(out_he, out_ihc, gt_keypoints, config=None, training_config=None, w_kp=1.0):
    """
    out_he / out_ihc: dicts from SuperPoint.forward(..., training=True)
    gt_keypoints:     list[Tensor(Ni, 3)] — (x, y, conf) HE CNN pixels
    returns:          (scalar loss tensor, dict of detached component tensors)
    """
    config = config or default_config
    training_config = training_config or TrainingConfig(name="default")
    kp_kwargs = {
        "cell_size":     default_config["grid_size"],
        "radius":        training_config.kp_radius,
        "w_loc":         training_config.w_loc,
        "w_fn":          training_config.w_fn,
        "w_fp":          training_config.w_fp,
        "match_mode":    training_config.match_mode,
        "match_epsilon": training_config.match_epsilon,
    }


    kp_he = utils.keypoint_matching_loss_detailed(out_he["logits"], gt_keypoints, **kp_kwargs)
    kp_ihc = utils.keypoint_matching_loss_detailed(out_ihc["logits"], gt_keypoints, **kp_kwargs)

    desc = utils.descriptor_loss(
        out_he["descriptors_raw"],
        out_ihc["descriptors_raw"],
        _desc_config(config, training_config),
    )

    loss_keypoint = kp_he["loss"] + kp_ihc["loss"]
    loss = w_kp * loss_keypoint + desc

    components = {
        "total": loss.detach(),
        "descriptor": desc.detach(),
        "keypoint": loss_keypoint.detach(),
        "loc": (kp_he["loss_loc"] + kp_ihc["loss_loc"]).detach(),
        "fn": (kp_he["loss_fn"] + kp_ihc["loss_fn"]).detach(),
        "fp": (kp_he["loss_fp"] + kp_ihc["loss_fp"]).detach(),
        "kp_he": kp_he["loss"].detach(),
        "kp_ihc": kp_ihc["loss"].detach(),
    }
    return loss, components


def _accumulate_losses(running, components):
    for key, value in components.items():
        running[key] = running.get(key, 0.0) + float(value)


def _mean_losses(running, count):
    return {key: value / count for key, value in running.items()}


@torch.no_grad()
def evaluate_kpis(model, loader, device, training_config):
    """
    Returns {"overall": {precision, recall, repeatability},
             "by_depth": {depth_str: {precision, recall, repeatability}, ...}}
    """
    model.eval()
    kp_kwargs = _match_kwargs(training_config)
    overall = _fresh_kpi_totals()
    by_depth: dict = {}

    for batch in loader:
        image_he  = batch["image_he"].to(device)
        image_ihc = batch["image_ihc"].to(device)
        gt        = [kp.to(device) for kp in batch["gt_keypoints"]]
        metas     = batch["meta"]

        out_he  = model({"image": image_he},  training=True)
        out_ihc = model({"image": image_ihc}, training=True)

        _accumulate_batch_kpis(overall, out_he["logits"], out_ihc["logits"], gt, kp_kwargs)

        for b in range(image_he.shape[0]):
            depth_key = metas[b]["depth"]
            _ensure_depth(by_depth, depth_key)
            match_he  = utils._match_keypoints_single(out_he["logits"][b],  gt[b], **kp_kwargs)
            match_ihc = utils._match_keypoints_single(out_ihc["logits"][b], gt[b], **kp_kwargs)
            d = by_depth[depth_key]
            d["repeatable"] += len(match_he["matched_gt_ids"] & match_ihc["matched_gt_ids"])
            d["total_gt"]   += match_he["num_gt"]
            d["tp"] += match_he["tp"] + match_ihc["tp"]
            d["fp"] += match_he["fp"] + match_ihc["fp"]
            d["fn"] += match_he["fn"] + match_ihc["fn"]

    return {
        "overall":  _kpis_from_totals(overall),
        "by_depth": {k: _kpis_from_totals(v) for k, v in sorted(by_depth.items())},
    }



def _make_train_loader(config, epoch, dataset=None):
    generator = torch.Generator()
    generator.manual_seed(epoch)
    return make_loader(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        dataset=dataset,
    )


def _print_epoch_progress(epoch, items_done, items_total):
    items_left = max(items_total - items_done, 0)
    print(f"epoch {epoch}: {items_done}/{items_total} done, {items_left} left", flush=True)


def _print_epoch_summary(epoch, train_means, train_kpis, duration_seconds, val_result=None):
    print(
        f"epoch {epoch} losses : "
        f"total={train_means['total']:.4f} "
        f"desc={train_means['descriptor']:.4f} "
        f"kp={train_means['keypoint']:.4f} "
        f"loc={train_means['loc']:.4f} "
        f"fn={train_means['fn']:.4f} "
        f"fp={train_means['fp']:.4f} "
        f"duration={duration_seconds:.1f}s",
        flush=True,
    )
    print(
        f"epoch {epoch} train  : "
        f"repeatability={train_kpis['repeatability']:.4f} "
        f"precision={train_kpis['precision']:.4f} "
        f"recall={train_kpis['recall']:.4f}",
        flush=True,
    )
    if val_result is not None:
        ov = val_result["overall"]
        print(
            f"epoch {epoch} val    : "
            f"repeatability={ov['repeatability']:.4f} "
            f"precision={ov['precision']:.4f} "
            f"recall={ov['recall']:.4f}",
            flush=True,
        )
        depth_parts = "  ".join(
            f"d{k}=[p={v['precision']:.3f} r={v['recall']:.3f}]"
            for k, v in val_result["by_depth"].items()
        )
        if depth_parts:
            print(f"epoch {epoch} val/d  : {depth_parts}", flush=True)


def train_epoch(model, loader, optimizer, device, training_config, epoch):
    model.train()
    running = {}
    batch_count = 0
    items_total = len(loader.dataset)
    samples_seen = 0
    epoch_totals = _fresh_kpi_totals()
    kp_kwargs = _match_kwargs(training_config)

    for batch in loader:
        batch_size = batch["image_he"].shape[0]

        optimizer.zero_grad(set_to_none=True)

        image_he = batch["image_he"].to(device)
        image_ihc = batch["image_ihc"].to(device)
        gt = [kp.to(device) for kp in batch["gt_keypoints"]]

        out_he = model({"image": image_he}, training=True)
        out_ihc = model({"image": image_ihc}, training=True)

        loss, components = total_loss(
            out_he,
            out_ihc,
            gt,
            training_config=training_config,
            w_kp=training_config.w_kp,
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _accumulate_batch_kpis(
                epoch_totals, out_he["logits"], out_ihc["logits"], gt, kp_kwargs
            )

        _accumulate_losses(running, components)
        batch_count += 1
        samples_seen += batch_size

        if batch_count % PROGRESS_EVERY_BATCHES == 0 or samples_seen >= items_total:
            _print_epoch_progress(epoch, samples_seen, items_total)

    if batch_count == 0:
        raise RuntimeError("train_epoch saw zero batches")

    return _mean_losses(running, batch_count), _kpis_from_totals(epoch_totals)


def save_checkpoint(model, instance, timestamp=None):
    """
    returns: (path, timestamp str) — path is run_dir/name_DD-MM_HH-MM.pth
    """
    timestamp = timestamp or checkpoint_timestamp()
    path = instance.run_dir / f"{instance.name}_{timestamp}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    instance.last_pth_path = path
    return path, timestamp


def _make_val_loader(config, dataset=None):
    dataset = dataset or StainPairKeypointDataset(split="val")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pairs,
    )


def train_model(instance, device=None, train_dataset=None, val_dataset=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = instance.config

    instance.run_dir.mkdir(parents=True, exist_ok=True)
    load_existing_run(instance)

    latest = latest_checkpoint_path(instance.run_dir, instance.name)
    if latest is not None:
        weights_path = latest
        instance.last_pth_path = latest
    else:
        weights_path = config.weights_init

    model = build_model(weights_path=weights_path, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = train_dataset or StainPairKeypointDataset(split="train")
    val_loader    = _make_val_loader(config, val_dataset)

    start_epoch = next_epoch_number(instance)
    end_epoch = start_epoch + config.num_epochs - 1

    interrupted = False
    current_epoch = None
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            current_epoch = epoch
            train_loader = _make_train_loader(config, epoch, dataset=train_dataset)
            try:
                epoch_start = time.perf_counter()
                train_means, train_kpis = train_epoch(
                    model, train_loader, optimizer, device, config, epoch
                )
                duration_seconds = time.perf_counter() - epoch_start
            except KeyboardInterrupt:
                interrupted = True
                break

            val_result = evaluate_kpis(model, val_loader, device, config)
            val_ov = val_result["overall"]

            log_entry = EpochLog(
                epoch=epoch,
                loss_total=train_means["total"],
                loss_descriptor=train_means["descriptor"],
                loss_keypoint=train_means["keypoint"],
                loss_loc=train_means["loc"],
                loss_fn=train_means["fn"],
                loss_fp=train_means["fp"],
                repeatability=train_kpis["repeatability"],
                precision=train_kpis["precision"],
                recall=train_kpis["recall"],
                duration_seconds=duration_seconds,
                val_precision=val_ov["precision"],
                val_recall=val_ov["recall"],
                val_repeatability=val_ov["repeatability"],
                val_kpis_by_depth=val_result["by_depth"],
            )
            instance.epoch_logs.append(log_entry)
            _print_epoch_summary(epoch, train_means, train_kpis, duration_seconds, val_result)

            if epoch % config.save_every_epochs == 0 or epoch == end_epoch:
                path, timestamp = save_checkpoint(model, instance)
                instance.save_log()
                print(f"epoch checkpoint saved : {path.name}  ts={timestamp}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if interrupted:
            path, timestamp = save_checkpoint(model, instance)
            instance.save_log()
            print(
                f"\ninterrupted during epoch {current_epoch}; "
                f"checkpoint saved {path.name} ts={timestamp}",
                flush=True,
            )

    if interrupted:
        raise KeyboardInterrupt

    return instance, model


if __name__ == "__main__":
    smoke_config = TrainingConfig(
        name="smoke",
        num_epochs=20,
        batch_size=32,
        num_workers=4,
        save_every_epochs=1,
    )
    instance = ModelInstance(
        name=smoke_config.name,
        config=smoke_config,
        parent="superpoint_v6_from_tf",
    )

    try:
        instance, model = train_model(instance)
    except KeyboardInterrupt:
        raise SystemExit(130)

    print(f"checkpoint : {instance.pth_path}")
    print(f"log        : {instance.log_path}")
