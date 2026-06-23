import shutil
import sys
import importlib
from pathlib import Path

import torch
import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PKG_ROOT.parent))
sys.path.append(str(PKG_ROOT))

import conf
importlib.reload(conf)

from superpoint_pytorch import SuperPoint
import utils
import training
from dataset import StainPairKeypointDataset, make_loader


CNN_H = conf.CNN_INPUT_HEIGHT
CNN_W = conf.CNN_INPUT_WIDTH
HC, WC = CNN_H // 8, CNN_W // 8


@pytest.fixture
def project_tmp():
    base = conf.PROJECT_ROOT / "introducing_superpoint" / "tests" / "_pytest_tmp"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def model():
    return training.build_model().eval()


def test_forward_inference_identical(model):
    torch.manual_seed(0)
    image = torch.rand(1, 1, CNN_H, CNN_W)

    with torch.no_grad():
        out_a = model({"image": image.clone()})
        out_b = model({"image": image.clone()})

    assert out_a.keys() == {"keypoints", "keypoint_scores", "descriptors"}
    assert torch.equal(out_a["keypoints"][0], out_b["keypoints"][0])
    assert torch.equal(out_a["descriptors"][0], out_b["descriptors"][0])


def test_forward_training_shapes(model):
    image = torch.rand(2, 1, CNN_H, CNN_W)
    out = model({"image": image}, training=True)

    assert set(out.keys()) == {"logits", "descriptors_raw"}
    assert tuple(out["logits"].shape) == (2, 65, HC, WC)
    assert tuple(out["descriptors_raw"].shape) == (2, 256, HC, WC)


def test_training_flag_does_not_alter_inference(model):
    torch.manual_seed(1)
    image = torch.rand(1, 1, CNN_H, CNN_W)
    with torch.no_grad():
        baseline = model({"image": image.clone()})
        _ = model({"image": image.clone()}, training=True)
        again = model({"image": image.clone()})
    assert torch.equal(baseline["keypoints"][0], again["keypoints"][0])


def _logits_peaked_at(coords, value=12.0):
    logits = torch.zeros(1, 65, HC, WC)
    logits[:, 64] = value
    for x, y in coords:
        cr = int(min(y // 8, HC - 1))
        cc = int(min(x // 8, WC - 1))
        br = int(y % 8)
        bc = int(x % 8)
        bin_idx = br * 8 + bc
        logits[0, 64, cr, cc] = 0.0
        logits[0, bin_idx, cr, cc] = value
    return logits


def test_keypoint_matching_loss_matches_reduce_false_negatives():
    coords = [(40, 32), (80, 80)]
    gt = [torch.tensor([[x, y, 1.0] for x, y in coords], dtype=torch.float32)]

    matched = _logits_peaked_at(coords)
    empty = torch.zeros(1, 65, HC, WC).index_fill_(1, torch.tensor([64]), 12.0)

    fn_matched = utils.keypoint_matching_loss(matched, gt, w_loc=0, w_fp=0)
    fn_empty = utils.keypoint_matching_loss(empty, gt, w_loc=0, w_fp=0)

    assert fn_matched.item() == pytest.approx(0.0, abs=1e-5)
    assert fn_empty.item() > fn_matched.item()


def test_keypoint_matching_loss_differentiable():
    coords = [(40, 32)]
    logits = _logits_peaked_at(coords).requires_grad_(True)
    gt = [torch.tensor([[40.0, 32.0, 1.0]])]
    loss = utils.keypoint_matching_loss(logits, gt)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(loss)


def test_dataset_item_shapes():
    dataset = StainPairKeypointDataset()
    assert len(dataset) > 0
    item = dataset[0]
    assert tuple(item["image_he"].shape) == (1, CNN_H, CNN_W)
    assert tuple(item["image_ihc"].shape) == (1, CNN_H, CNN_W)
    assert item["gt_keypoints"].dim() == 2 and item["gt_keypoints"].shape[1] == 3


def test_make_eval_loader_uses_fixed_subset():
    from model_instance import TrainingConfig

    config = TrainingConfig(name="eval_subset", eval_num_samples=17, eval_seed=3)
    dataset = StainPairKeypointDataset(split="val")
    loader = training._make_eval_loader(config, dataset)
    assert len(loader.dataset) == min(17, len(dataset))


def test_pipeline_smoke_backward(model):
    model.train()
    loader = make_loader(batch_size=2, shuffle=False)
    batch = next(iter(loader))

    out_he = model({"image": batch["image_he"]}, training=True)
    out_ihc = model({"image": batch["image_ihc"]}, training=True)

    loss, components = training.total_loss(out_he, out_ihc, batch["gt_keypoints"])

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert "loc" in components and "fn" in components and "fp" in components
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_compute_keypoint_kpis_perfect_match():
    coords = [(40, 32), (80, 80)]
    gt = [torch.tensor([[x, y, 1.0] for x, y in coords], dtype=torch.float32)]
    logits = _logits_peaked_at(coords)
    kpis = utils.compute_keypoint_kpis(logits, logits, gt)
    assert kpis["precision"] == pytest.approx(1.0, abs=1e-4)
    assert kpis["recall"] == pytest.approx(1.0, abs=1e-4)
    assert kpis["repeatability"] == pytest.approx(1.0, abs=1e-4)


def test_model_instance_log_roundtrip(project_tmp):
    from model_instance import EpochLog, EvaluationLog, ModelInstance, ResumeState, TrainingConfig

    config = TrainingConfig(
        name="unit",
        run_dir=project_tmp,
        num_epochs=1,
        save_every_epochs=1,
        eval_every_seconds=1800,
        eval_num_samples=128,
        eval_seed=4,
        eval_batch_size=8,
    )
    instance = ModelInstance(
        name="unit",
        config=config,
        parent="parent_run",
        epoch_logs=[
            EpochLog(
                epoch=1,
                loss_total=1.0,
                loss_descriptor=0.1,
                loss_keypoint=0.8,
                loss_loc=0.2,
                loss_fn=0.3,
                loss_fp=0.1,
                repeatability=0.5,
                precision=0.6,
                recall=0.7,
            )
        ],
        evaluation_logs=[
            EvaluationLog(
                timestamp="2026-06-23T19:12:04",
                epoch=1,
                batch_idx=12,
                samples_seen=48,
                duration_seconds=7.5,
                num_batches=4,
                num_samples=16,
                precision=0.6,
                recall=0.7,
                repeatability=0.5,
                kpis_by_depth={"d5": {"precision": 0.6, "recall": 0.7, "repeatability": 0.5}},
            )
        ],
        resume_state=ResumeState(epoch=2, next_batch_idx=7),
    )
    log_path = instance.save_log()
    restored = ModelInstance.load_log(log_path)
    assert restored.name == "unit"
    assert restored.parent == "parent_run"
    assert restored.config.eval_every_seconds == 1800
    assert restored.config.eval_num_samples == 128
    assert restored.config.eval_seed == 4
    assert restored.config.eval_batch_size == 8
    assert len(restored.epoch_logs) == 1
    assert len(restored.evaluation_logs) == 1
    assert restored.epoch_logs[0].recall == pytest.approx(0.7)
    assert restored.evaluation_logs[0].timestamp == "2026-06-23T19:12:04"
    assert restored.evaluation_logs[0].duration_seconds == pytest.approx(7.5)
    assert restored.resume_state.epoch == 2
    assert restored.resume_state.next_batch_idx == 7


def test_train_model_writes_checkpoint_and_log(project_tmp):
    from model_instance import ModelInstance, TrainingConfig

    config = TrainingConfig(
        name="mini_train",
        run_dir=project_tmp,
        num_epochs=1,
        batch_size=2,
        save_every_epochs=1
    )
    instance = ModelInstance(
        name=config.name,
        config=config,
        parent="superpoint_v6_from_tf",
    )
    instance, _ = training.train_model(instance)
    assert instance.pth_path.parent == project_tmp / "mini_train"
    assert instance.pth_path.name.startswith("mini_train_")
    assert instance.pth_path.exists()
    assert instance.log_path.exists()
    assert len(instance.epoch_logs) == 1
    last = instance.epoch_logs[0]
    assert last.epoch == 1
    assert 0.0 <= last.repeatability <= 1.0
    assert 0.0 <= last.precision <= 1.0
    assert 0.0 <= last.recall <= 1.0


def test_load_existing_run_restores_logs(project_tmp):
    from model_instance import EpochLog, ModelInstance, ResumeState, TrainingConfig, load_existing_run

    config = TrainingConfig(name="resume_log", run_dir=project_tmp, num_epochs=5)
    instance = ModelInstance(
        name=config.name,
        config=config,
        epoch_logs=[
            EpochLog(
                epoch=100,
                loss_total=1.0,
                loss_descriptor=0.1,
                loss_keypoint=0.8,
                loss_loc=0.2,
                loss_fn=0.3,
                loss_fp=0.1,
                repeatability=0.5,
                precision=0.6,
                recall=0.7,
            )
        ],
        resume_state=ResumeState(epoch=101, next_batch_idx=3),
    )
    instance.save_log()

    fresh = ModelInstance(name=config.name, config=config)
    assert load_existing_run(fresh)
    assert len(fresh.epoch_logs) == 1
    assert fresh.epoch_logs[0].epoch == 100
    assert fresh.resume_state.epoch == 101
    assert fresh.resume_state.next_batch_idx == 3


def test_latest_checkpoint_path_picks_newest(project_tmp):
    import os
    from model_instance import latest_checkpoint_path

    run_dir = project_tmp / "ckpt"
    run_dir.mkdir(parents=True)
    older = run_dir / "ckpt_01-06_10-00.pth"
    newer = run_dir / "ckpt_02-06_10-00.pth"
    older.touch()
    newer.touch()
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert latest_checkpoint_path(run_dir, "ckpt").resolve() == newer.resolve()


def test_next_epoch_number(project_tmp):
    from model_instance import EpochLog, ModelInstance, TrainingConfig, next_epoch_number

    config = TrainingConfig(name="n", run_dir=project_tmp)
    empty = ModelInstance(name="n", config=config)
    assert next_epoch_number(empty) == 1

    logged = ModelInstance(
        name="n",
        config=config,
        epoch_logs=[EpochLog(epoch=100, loss_total=0, loss_descriptor=0, loss_keypoint=0,
                             loss_loc=0, loss_fn=0, loss_fp=0, repeatability=0, precision=0, recall=0)],
    )
    assert next_epoch_number(logged) == 101


def test_train_model_appends_epochs(project_tmp, model, monkeypatch):
    from model_instance import EpochLog, ModelInstance, TrainingConfig, next_epoch_number

    config = TrainingConfig(name="append", run_dir=project_tmp, num_epochs=2, batch_size=2)
    run_dir = project_tmp / "append"
    run_dir.mkdir(parents=True)

    existing = ModelInstance(
        name=config.name,
        config=config,
        epoch_logs=[
            EpochLog(
                epoch=100,
                loss_total=1.0,
                loss_descriptor=0.1,
                loss_keypoint=0.8,
                loss_loc=0.2,
                loss_fn=0.3,
                loss_fp=0.1,
                repeatability=0.5,
                precision=0.6,
                recall=0.7,
            )
        ],
    )
    existing.save_log()

    seen_epochs = []

    def fake_train_epoch(
        model, loader, optimizer, device, training_config, epoch, start_batch_idx=0,
        items_total=None, instance=None, eval_loader=None,
    ):
        seen_epochs.append(epoch)
        return {"total": 1.0, "descriptor": 0.1, "keypoint": 0.8, "loc": 0.2, "fn": 0.3, "fp": 0.1}, {
            "repeatability": 0.5, "precision": 0.6, "recall": 0.7,
        }

    monkeypatch.setattr(training, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: model())

    instance = ModelInstance(name=config.name, config=config)
    training.train_model(instance)

    assert seen_epochs == [101, 102]
    assert len(instance.epoch_logs) == 3
    assert instance.epoch_logs[-1].epoch == 102
    assert next_epoch_number(instance) == 103


def test_train_model_resumes_mid_epoch_and_clears_state(project_tmp, model, monkeypatch):
    from model_instance import EpochLog, ModelInstance, ResumeState, TrainingConfig

    config = TrainingConfig(name="mid_epoch", run_dir=project_tmp, num_epochs=1, batch_size=2)
    existing = ModelInstance(
        name=config.name,
        config=config,
        epoch_logs=[
            EpochLog(
                epoch=100,
                loss_total=1.0,
                loss_descriptor=0.1,
                loss_keypoint=0.8,
                loss_loc=0.2,
                loss_fn=0.3,
                loss_fp=0.1,
                repeatability=0.5,
                precision=0.6,
                recall=0.7,
            )
        ],
        resume_state=ResumeState(epoch=101, next_batch_idx=4),
    )
    existing.save_log()

    calls = []

    def fake_train_epoch(
        model, loader, optimizer, device, training_config, epoch, start_batch_idx=0,
        items_total=None, instance=None, eval_loader=None,
    ):
        calls.append((epoch, start_batch_idx, items_total, len(loader.dataset)))
        return {"total": 1.0, "descriptor": 0.1, "keypoint": 0.8, "loc": 0.2, "fn": 0.3, "fp": 0.1}, {
            "repeatability": 0.5, "precision": 0.6, "recall": 0.7,
        }

    monkeypatch.setattr(training, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(training, "build_model", lambda *args, **kwargs: model())
    monkeypatch.setattr(
        training,
        "evaluate_kpis",
        lambda *args, **kwargs: {
            "overall": {"precision": 0.6, "recall": 0.7, "repeatability": 0.5},
            "by_depth": {},
        },
    )

    instance = ModelInstance(name=config.name, config=config)
    training.train_model(instance)

    train_len = len(StainPairKeypointDataset(split="train"))
    assert calls == [(101, 4, train_len, train_len - 8)]
    assert instance.resume_state is None
    restored = ModelInstance.load_log(instance.log_path)
    assert restored.resume_state is None
    assert restored.epoch_logs[-1].epoch == 101


def test_train_model_saves_on_keyboard_interrupt(project_tmp, monkeypatch):
    from model_instance import ModelInstance, TrainingConfig

    config = TrainingConfig(
        name="interrupt",
        run_dir=project_tmp,
        num_epochs=3,
        batch_size=2,
        save_every_epochs=5
    )
    instance = ModelInstance(name=config.name, config=config)

    def raise_interrupt(*args, **kwargs):
        raise training.MidEpochInterrupt(epoch=1, next_batch_idx=3)

    monkeypatch.setattr(training, "train_epoch", raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        training.train_model(instance)
    assert instance.pth_path.exists()
    assert instance.log_path.exists()
    assert len(instance.epoch_logs) == 0
    assert instance.resume_state.epoch == 1
    assert instance.resume_state.next_batch_idx == 3


def test_checkpoint_timestamp_format():
    from datetime import datetime
    from model_instance import checkpoint_timestamp

    ts = checkpoint_timestamp(datetime(2026, 6, 12, 14, 35))
    assert ts == "12-06_14-35"


def test_explore_dataset_render_and_save(project_tmp):
    import matplotlib
    matplotlib.use("Agg")

    import explore_dataset

    dataset = StainPairKeypointDataset()
    item = dataset[0]
    arr = explore_dataset.tensor_to_gray_numpy(item["image_he"])
    assert arr.shape == (CNN_H, CNN_W)
    assert arr.min() >= 0.0 and arr.max() <= 1.0

    out_path = project_tmp / "pair.png"
    explore_dataset.save_training_pair(item, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    fig = explore_dataset.render_training_grid(dataset, indices=[0, 1], ncols=2)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
