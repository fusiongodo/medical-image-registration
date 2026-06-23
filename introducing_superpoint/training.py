"""
SuperPoint stain-invariant training: model load, loss assembly, epoch loop,
checkpointing, and KPI logging via ModelInstance.
"""
import sys
import time
import importlib
from datetime import datetime
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
import conf
importlib.reload(conf)

from superpoint_pytorch import SuperPoint, default_config
from model_instance import (
    EpochLog,
    EvaluationLog,
    ModelInstance,
    ResumeState,
    TrainingConfig,
    checkpoint_timestamp,
    load_existing_run,
    latest_checkpoint_path,
    next_epoch_number,
)
from torch.utils.data import DataLoader, Subset
from dataset import StainPairKeypointDataset, collate_pairs, make_loader
import utils

DEFAULT_WEIGHTS = conf.resolve("introducing_superpoint/superpoint_v6_from_tf.pth")
PROGRESS_EVERY_BATCHES = 10


class MidEpochInterrupt(Exception):
    def __init__(self, epoch: int, next_batch_idx: int):
        super().__init__(f"interrupted during epoch {epoch} at batch {next_batch_idx}")
        self.epoch = epoch
        self.next_batch_idx = next_batch_idx


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


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
    was_training = model.training
    start = time.perf_counter()
    model.eval()
    kp_kwargs = _match_kwargs(training_config)
    overall = _fresh_kpi_totals()
    by_depth: dict = {}
    num_batches = 0
    num_samples = 0

    for batch in loader:
        image_he  = batch["image_he"].to(device)
        image_ihc = batch["image_ihc"].to(device)
        gt        = [kp.to(device) for kp in batch["gt_keypoints"]]
        metas     = batch["meta"]
        num_batches += 1
        num_samples += image_he.shape[0]

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

    result = {
        "overall":  _kpis_from_totals(overall),
        "by_depth": {k: _kpis_from_totals(v) for k, v in sorted(by_depth.items())},
        "duration_seconds": time.perf_counter() - start,
        "num_batches": num_batches,
        "num_samples": num_samples,
    }
    if was_training:
        model.train()
    return result



def _make_train_loader(config, epoch, dataset=None, start_batch_idx=0):
    dataset = dataset or StainPairKeypointDataset(split="train")
    generator = torch.Generator()
    generator.manual_seed(epoch)
    shuffle = True
    if start_batch_idx > 0:
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        start_sample_idx = min(start_batch_idx * config.batch_size, len(indices))
        dataset = Subset(dataset, indices[start_sample_idx:])
        shuffle = False
        generator = None
    return make_loader(
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        dataset=dataset,
    )


def _make_eval_loader(config, dataset=None):
    dataset = dataset or StainPairKeypointDataset(split="val")
    generator = torch.Generator()
    generator.manual_seed(config.eval_seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    indices = indices[:min(config.eval_num_samples, len(indices))]
    dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pairs,
    )


def _print_epoch_progress(epoch, items_done, items_total, elapsed_seconds=None, window_items=None):
    items_left = max(items_total - items_done, 0)
    extra = ""
    if elapsed_seconds is not None and window_items is not None:
        extra = f", last_{window_items}={elapsed_seconds:.1f}s"
    _log(f"epoch {epoch}: {items_done}/{items_total} done, {items_left} left{extra}")


def _print_epoch_summary(epoch, train_means, train_kpis, duration_seconds, val_result=None):
    _log(
        f"epoch {epoch} losses : "
        f"total={train_means['total']:.4f} "
        f"desc={train_means['descriptor']:.4f} "
        f"kp={train_means['keypoint']:.4f} "
        f"loc={train_means['loc']:.4f} "
        f"fn={train_means['fn']:.4f} "
        f"fp={train_means['fp']:.4f} "
        f"duration={duration_seconds:.1f}s"
    )
    _log(
        f"epoch {epoch} train  : "
        f"repeatability={train_kpis['repeatability']:.4f} "
        f"precision={train_kpis['precision']:.4f} "
        f"recall={train_kpis['recall']:.4f}"
    )
    if val_result is not None:
        ov = val_result["overall"]
        _log(
            f"epoch {epoch} val    : "
            f"repeatability={ov['repeatability']:.4f} "
            f"precision={ov['precision']:.4f} "
            f"recall={ov['recall']:.4f}"
        )
        depth_parts = "  ".join(
            f"d{k}=[p={v['precision']:.3f} r={v['recall']:.3f}]"
            for k, v in val_result["by_depth"].items()
        )
        if depth_parts:
            _log(f"epoch {epoch} val/d  : {depth_parts}")


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    training_config,
    epoch,
    start_batch_idx=0,
    items_total=None,
    instance=None,
    eval_loader=None,
):
    model.train()
    running = {}
    batch_count = 0
    items_total = items_total or len(loader.dataset)
    samples_seen = min(start_batch_idx * training_config.batch_size, items_total)
    epoch_totals = _fresh_kpi_totals()
    kp_kwargs = _match_kwargs(training_config)
    next_batch_idx = start_batch_idx
    next_eval_at = time.monotonic() + training_config.eval_every_seconds
    last_progress_at = time.monotonic()
    last_progress_samples = samples_seen

    try:
        for batch_idx, batch in enumerate(loader):
            batch_size = batch["image_he"].shape[0]
            original_batch_idx = start_batch_idx + batch_idx

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
            next_batch_idx = original_batch_idx + 1

            with torch.no_grad():
                _accumulate_batch_kpis(
                    epoch_totals, out_he["logits"], out_ihc["logits"], gt, kp_kwargs
                )

            _accumulate_losses(running, components)
            batch_count += 1
            samples_seen += batch_size

            if batch_count % PROGRESS_EVERY_BATCHES == 0 or samples_seen >= items_total:
                now = time.monotonic()
                window_items = samples_seen - last_progress_samples
                _print_epoch_progress(
                    epoch, samples_seen, items_total, now - last_progress_at, window_items
                )
                last_progress_at = now
                last_progress_samples = samples_seen

            if (
                instance is not None
                and eval_loader is not None
                and training_config.eval_every_seconds > 0
                and time.monotonic() >= next_eval_at
            ):
                eval_result = evaluate_kpis(model, eval_loader, device, training_config)
                overall = eval_result["overall"]
                evaluation_log = EvaluationLog(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    epoch=epoch,
                    batch_idx=next_batch_idx,
                    samples_seen=samples_seen,
                    duration_seconds=eval_result["duration_seconds"],
                    num_batches=eval_result["num_batches"],
                    num_samples=eval_result["num_samples"],
                    precision=overall["precision"],
                    recall=overall["recall"],
                    repeatability=overall["repeatability"],
                    kpis_by_depth=eval_result["by_depth"],
                )
                instance.evaluation_logs.append(evaluation_log)
                instance.save_log()
                _log(
                    f"eval epoch={epoch} batch={next_batch_idx} "
                    f"train_samples={samples_seen} eval_samples={evaluation_log.num_samples} "
                    f"duration={evaluation_log.duration_seconds:.1f}s "
                    f"precision={evaluation_log.precision:.4f} "
                    f"recall={evaluation_log.recall:.4f} "
                    f"repeatability={evaluation_log.repeatability:.4f}"
                )
                next_eval_at = time.monotonic() + training_config.eval_every_seconds
                model.train()
    except KeyboardInterrupt as exc:
        raise MidEpochInterrupt(epoch, next_batch_idx) from exc

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
    eval_loader = _make_eval_loader(config, val_dataset)

    resume_state = instance.resume_state
    start_epoch = resume_state.epoch if resume_state else next_epoch_number(instance)
    end_epoch = start_epoch + config.num_epochs - 1

    interrupted = False
    current_epoch = None
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            current_epoch = epoch
            try:
                epoch_start = time.perf_counter()
                start_batch_idx = (
                    resume_state.next_batch_idx
                    if resume_state is not None and epoch == resume_state.epoch
                    else 0
                )
                train_loader = _make_train_loader(
                    config, epoch, dataset=train_dataset, start_batch_idx=start_batch_idx
                )
                train_means, train_kpis = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    config,
                    epoch,
                    start_batch_idx,
                    len(train_dataset),
                    instance,
                    eval_loader,
                )
                duration_seconds = time.perf_counter() - epoch_start
            except MidEpochInterrupt as exc:
                interrupted = True
                instance.resume_state = ResumeState(
                    epoch=exc.epoch,
                    next_batch_idx=exc.next_batch_idx,
                )
                break
            except KeyboardInterrupt:
                interrupted = True
                break

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
            )
            instance.epoch_logs.append(log_entry)
            instance.resume_state = None
            resume_state = None
            _print_epoch_summary(epoch, train_means, train_kpis, duration_seconds)

            if epoch % config.save_every_epochs == 0 or epoch == end_epoch:
                path, timestamp = save_checkpoint(model, instance)
                instance.save_log()
                _log(f"epoch checkpoint saved : {path.name}  ts={timestamp}")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if interrupted:
            path, timestamp = save_checkpoint(model, instance)
            instance.save_log()
            _log(
                f"interrupted during epoch {current_epoch}; "
                f"checkpoint saved {path.name} ts={timestamp}"
            )

    if interrupted:
        raise KeyboardInterrupt

    return instance, model


if __name__ == "__main__":
    oyster_config = TrainingConfig(
        name="oyster",
        num_epochs=20,
        batch_size=4,
        num_workers=2,
        save_every_epochs=1,
    )
    instance = ModelInstance(
        name=oyster_config.name,
        config=oyster_config,
        parent="superpoint_v6_from_tf",
    )

    try:
        instance, model = train_model(instance)
    except KeyboardInterrupt:
        raise SystemExit(130)

    _log(f"checkpoint : {instance.pth_path}")
    _log(f"log        : {instance.log_path}")
