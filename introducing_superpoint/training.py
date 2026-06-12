"""
SuperPoint stain-invariant training: model load, loss assembly, epoch loop,
checkpointing, and KPI logging via ModelInstance.
"""
import sys
import importlib
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
import conf
importlib.reload(conf)

from superpoint_pytorch import SuperPoint, default_config
from model_instance import (
    CheckpointLog,
    EpochLog,
    ModelInstance,
    TrainingConfig,
    checkpoint_timestamp,
    should_resume_training,
)
from dataset import StainPairKeypointDataset, make_loader
import utils

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "superpoint_v6_from_tf.pth"
PROGRESS_EVERY_BATCHES = 10


def build_model(weights_path=DEFAULT_WEIGHTS, device="cpu"):
    model = SuperPoint()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def _kp_kwargs(config):
    return {
        "cell_size": default_config["grid_size"],
        "radius": config.kp_radius,
        "w_loc": config.w_loc,
        "w_fn": config.w_fn,
        "w_fp": config.w_fp,
    }


def _match_kwargs(config):
    return {
        "cell_size": default_config["grid_size"],
        "radius": config.kp_radius,
    }


def _fresh_kpi_totals():
    return {
        "repeatable": 0,
        "total_gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def _fresh_gt_bin_totals():
    return {"correct": 0, "total_gt": 0}


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
    kp_kwargs = _kp_kwargs(training_config)

    kp_he = utils.keypoint_matching_loss_detailed(out_he["logits"], gt_keypoints, **kp_kwargs)
    kp_ihc = utils.keypoint_matching_loss_detailed(out_ihc["logits"], gt_keypoints, **kp_kwargs)

    batch_size = out_he["logits"].shape[0]
    identity = torch.eye(3, device=out_he["logits"].device).unsqueeze(0).expand(batch_size, -1, -1)

    desc = utils.descriptor_loss(
        out_he["descriptors_raw"],
        out_ihc["descriptors_raw"],
        identity,
        config,
        warp_points=utils.warp_points,
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
    model.eval()
    kp_kwargs = _match_kwargs(training_config)
    totals = _fresh_kpi_totals()

    for batch_idx, batch in enumerate(loader):
        if training_config.max_batches_per_epoch is not None:
            if batch_idx >= training_config.max_batches_per_epoch:
                break

        image_he = batch["image_he"].to(device)
        image_ihc = batch["image_ihc"].to(device)
        gt = [kp.to(device) for kp in batch["gt_keypoints"]]

        out_he = model({"image": image_he}, training=True)
        out_ihc = model({"image": image_ihc}, training=True)
        _accumulate_batch_kpis(totals, out_he["logits"], out_ihc["logits"], gt, kp_kwargs)

    return _kpis_from_totals(totals)


def _items_in_epoch(dataset_len, training_config):
    if training_config.max_batches_per_epoch is not None:
        return min(
            dataset_len,
            training_config.max_batches_per_epoch * training_config.batch_size,
        )
    return dataset_len


def _make_train_loader(config, epoch, dataset=None):
    generator = torch.Generator()
    generator.manual_seed(epoch)
    return make_loader(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
        dataset=dataset,
    )


def _print_epoch_progress(epoch, items_done, items_total):
    items_left = max(items_total - items_done, 0)
    print(f"epoch {epoch}: {items_done}/{items_total} done, {items_left} left", flush=True)


def _print_kpi_checkpoint(epoch, items_done, kpis, gt_bin_recall):
    print(
        f"epoch {epoch} @{items_done} KPIs : "
        f"repeatability={kpis['repeatability']:.4f} "
        f"precision={kpis['precision']:.4f} "
        f"recall={kpis['recall']:.4f} "
        f"gt_bin_recall={gt_bin_recall:.4f} "
        f"tp={kpis['tp']} fp={kpis['fp']} fn={kpis['fn']}",
        flush=True,
    )


def _print_epoch_summary(epoch, train_means, kpis, gt_bin_recall):
    print(
        f"epoch {epoch} losses : "
        f"total={train_means['total']:.4f} "
        f"desc={train_means['descriptor']:.4f} "
        f"kp={train_means['keypoint']:.4f} "
        f"loc={train_means['loc']:.4f} "
        f"fn={train_means['fn']:.4f} "
        f"fp={train_means['fp']:.4f}",
        flush=True,
    )
    print(
        f"epoch {epoch} KPIs   : "
        f"repeatability={kpis['repeatability']:.4f} "
        f"precision={kpis['precision']:.4f} "
        f"recall={kpis['recall']:.4f} "
        f"gt_bin_recall={gt_bin_recall:.4f}",
        flush=True,
    )


def _maybe_checkpoint_window(
    model,
    instance,
    epoch,
    samples_seen,
    last_checkpoint_at,
    window_totals,
    window_gt_bin_totals,
    window_running,
    window_batch_count,
    training_config,
):
    if samples_seen - last_checkpoint_at < training_config.kpi_every_instances:
        return last_checkpoint_at, window_totals, window_gt_bin_totals, window_running, window_batch_count

    kpis = _kpis_from_totals(window_totals)
    kpis["tp"] = window_totals["tp"]
    kpis["fp"] = window_totals["fp"]
    kpis["fn"] = window_totals["fn"]
    gt_bin_recall = utils.gt_bin_recall_from_totals(window_gt_bin_totals)
    window_means = _mean_losses(window_running, max(window_batch_count, 1))

    _print_kpi_checkpoint(epoch, samples_seen, kpis, gt_bin_recall)

    path, timestamp = save_checkpoint(model, instance)
    instance.checkpoint_logs.append(
        CheckpointLog(
            epoch=epoch,
            sample_idx=samples_seen,
            timestamp=timestamp,
            pth_path=str(path),
            repeatability=kpis["repeatability"],
            precision=kpis["precision"],
            recall=kpis["recall"],
            gt_bin_recall=gt_bin_recall,
            tp=window_totals["tp"],
            fp=window_totals["fp"],
            fn=window_totals["fn"],
            total_gt=window_totals["total_gt"],
            repeatable=window_totals["repeatable"],
            loss_total=window_means.get("total", 0.0),
            loss_keypoint=window_means.get("keypoint", 0.0),
            loss_descriptor=window_means.get("descriptor", 0.0),
        )
    )
    log_path = instance.save_log()
    print(
        f"checkpoint saved @{samples_seen} : {path.name}  log={log_path.name}  ts={timestamp}",
        flush=True,
    )

    return samples_seen, _fresh_kpi_totals(), _fresh_gt_bin_totals(), {}, 0


def train_epoch(model, loader, optimizer, device, training_config, instance, epoch):
    model.train()
    running = {}
    batch_count = 0
    items_total = _items_in_epoch(len(loader.dataset), training_config)
    start_idx = instance.resume_sample_idx if epoch == instance.resume_epoch else 0
    samples_seen = start_idx
    last_checkpoint_at = start_idx
    window_totals = _fresh_kpi_totals()
    window_gt_bin_totals = _fresh_gt_bin_totals()
    window_running = {}
    window_batch_count = 0
    epoch_totals = _fresh_kpi_totals()
    epoch_gt_bin_totals = _fresh_gt_bin_totals()
    kp_kwargs = _match_kwargs(training_config)

    if start_idx >= items_total:
        instance.resume_sample_idx = 0
        instance.resume_epoch = epoch + 1
        return None

    try:
        for batch_idx, batch in enumerate(loader):
            if training_config.max_batches_per_epoch is not None:
                if batch_idx >= training_config.max_batches_per_epoch:
                    break

            batch_size = batch["image_he"].shape[0]
            if samples_seen + batch_size <= start_idx:
                samples_seen += batch_size
                continue

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
                    window_totals, out_he["logits"], out_ihc["logits"], gt, kp_kwargs
                )
                _accumulate_batch_kpis(
                    epoch_totals, out_he["logits"], out_ihc["logits"], gt, kp_kwargs
                )
                utils.accumulate_gt_bin_recall(
                    window_gt_bin_totals,
                    out_he["logits"],
                    out_ihc["logits"],
                    gt,
                    cell_size=default_config["grid_size"],
                )
                utils.accumulate_gt_bin_recall(
                    epoch_gt_bin_totals,
                    out_he["logits"],
                    out_ihc["logits"],
                    gt,
                    cell_size=default_config["grid_size"],
                )

            _accumulate_losses(running, components)
            _accumulate_losses(window_running, components)
            batch_count += 1
            window_batch_count += 1
            samples_seen += batch_size

            instance.resume_epoch = epoch
            instance.resume_sample_idx = samples_seen

            items_done = min(samples_seen, items_total)
            if batch_count % PROGRESS_EVERY_BATCHES == 0 or items_done >= items_total:
                _print_epoch_progress(epoch, items_done, items_total)

            (
                last_checkpoint_at,
                window_totals,
                window_gt_bin_totals,
                window_running,
                window_batch_count,
            ) = _maybe_checkpoint_window(
                model,
                instance,
                epoch,
                samples_seen,
                last_checkpoint_at,
                window_totals,
                window_gt_bin_totals,
                window_running,
                window_batch_count,
                training_config,
            )
    except KeyboardInterrupt:
        raise

    if batch_count == 0:
        raise RuntimeError("train_epoch saw zero batches")

    instance.resume_sample_idx = 0
    instance.resume_epoch = epoch + 1
    epoch_kpis = _kpis_from_totals(epoch_totals)
    epoch_kpis["gt_bin_recall"] = utils.gt_bin_recall_from_totals(epoch_gt_bin_totals)
    return _mean_losses(running, batch_count), epoch_kpis


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


def train_model(instance, device=None, train_dataset=None, eval_loader=None, resume=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = instance.config

    if resume is None:
        resume = should_resume_training(instance)

    if resume:
        instance.merge_saved_state()
    else:
        instance.resume_epoch = 1
        instance.resume_sample_idx = 0
        instance.epoch_logs = []
        instance.checkpoint_logs = []

    if instance.last_pth_path is not None and instance.last_pth_path.exists():
        weights_path = instance.last_pth_path
    elif instance.pth_path.exists():
        weights_path = instance.pth_path
    else:
        weights_path = config.weights_init
    model = build_model(weights_path=weights_path, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = train_dataset or StainPairKeypointDataset()

    instance.run_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = instance.resume_epoch

    interrupted = False
    try:
        for epoch in range(start_epoch, config.num_epochs + 1):
            train_loader = _make_train_loader(config, epoch, dataset=train_dataset)
            try:
                result = train_epoch(
                    model, train_loader, optimizer, device, config, instance, epoch
                )
            except KeyboardInterrupt:
                interrupted = True
                break

            if result is None:
                continue

            train_means, kpis = result

            log_entry = EpochLog(
                epoch=epoch,
                loss_total=train_means["total"],
                loss_descriptor=train_means["descriptor"],
                loss_keypoint=train_means["keypoint"],
                loss_loc=train_means["loc"],
                loss_fn=train_means["fn"],
                loss_fp=train_means["fp"],
                repeatability=kpis["repeatability"],
                precision=kpis["precision"],
                recall=kpis["recall"],
                gt_bin_recall=kpis["gt_bin_recall"],
            )
            instance.epoch_logs.append(log_entry)
            _print_epoch_summary(epoch, train_means, kpis, kpis["gt_bin_recall"])

            if epoch % config.save_every_epochs == 0 or epoch == config.num_epochs:
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
                f"\ninterrupted at epoch {instance.resume_epoch}, "
                f"sample {instance.resume_sample_idx}; "
                f"checkpoint saved {path.name} ts={timestamp}",
                flush=True,
            )

    if interrupted:
        raise KeyboardInterrupt

    return instance, model


if __name__ == "__main__":
    smoke_config = TrainingConfig(
        name="smoke",
        num_epochs=1,
        batch_size=2,
        save_every_epochs=1,
        max_batches_per_epoch=None,
        kpi_every_instances=720,
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
