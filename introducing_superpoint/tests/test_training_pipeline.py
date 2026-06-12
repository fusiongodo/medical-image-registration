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


def test_warp_points_identity():
    pts = torch.tensor([[10.0, 5.0], [0.0, 0.0], [3.5, 7.25]])
    identity = torch.eye(3)
    warped = utils.warp_points(pts, identity)
    assert torch.allclose(warped, pts, atol=1e-6)


def test_warp_points_translation():
    pts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    H = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    warped = utils.warp_points(pts, H)
    expected = pts + torch.tensor([5.0, -2.0])
    assert torch.allclose(warped, expected, atol=1e-6)


def test_warp_points_batched():
    pts = torch.tensor([[1.0, 1.0]])
    H = torch.stack([torch.eye(3), torch.eye(3) * 2.0])
    H[1, 2, 2] = 1.0
    warped = utils.warp_points(pts, H)
    assert tuple(warped.shape) == (2, 1, 2)
    assert torch.allclose(warped[0, 0], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert torch.allclose(warped[1, 0], torch.tensor([2.0, 2.0]), atol=1e-6)


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


def test_model_instance_log_roundtrip(tmp_path):
    from model_instance import EpochLog, ModelInstance, TrainingConfig

    config = TrainingConfig(
        name="unit",
        run_dir=tmp_path,
        num_epochs=1,
        save_every_epochs=1,
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
                gt_bin_recall=0.8,
            )
        ],
    )
    log_path = instance.save_log()
    restored = ModelInstance.load_log(log_path)
    assert restored.name == "unit"
    assert restored.parent == "parent_run"
    assert restored.resume_epoch == 1
    assert restored.resume_sample_idx == 0
    assert len(restored.epoch_logs) == 1
    assert restored.epoch_logs[0].recall == pytest.approx(0.7)


def test_train_model_writes_checkpoint_and_log(tmp_path):
    from model_instance import ModelInstance, TrainingConfig

    config = TrainingConfig(
        name="mini_train",
        run_dir=tmp_path,
        num_epochs=1,
        batch_size=2,
        save_every_epochs=1,
        max_batches_per_epoch=1,
    )
    instance = ModelInstance(
        name=config.name,
        config=config,
        parent="superpoint_v6_from_tf",
    )
    instance, _ = training.train_model(instance)
    assert instance.pth_path.parent == tmp_path / "mini_train"
    assert instance.pth_path.name.startswith("mini_train_")
    assert instance.pth_path.exists()
    assert instance.log_path.exists()
    assert len(instance.epoch_logs) == 1
    last = instance.epoch_logs[0]
    assert last.epoch == 1
    assert 0.0 <= last.repeatability <= 1.0
    assert 0.0 <= last.precision <= 1.0
    assert 0.0 <= last.recall <= 1.0


def test_resume_sample_idx_skips_processed_batches(tmp_path):
    from model_instance import ModelInstance, TrainingConfig

    class TinyDataset:
        def __len__(self):
            return 6

        def __getitem__(self, idx):
            image = torch.zeros(1, conf.CNN_INPUT_HEIGHT, conf.CNN_INPUT_WIDTH)
            return {
                "image_he": image,
                "image_ihc": image,
                "gt_keypoints": torch.tensor([[10.0, 10.0, 1.0]]),
                "meta": {"idx": idx},
            }

    config = TrainingConfig(
        name="resume",
        run_dir=tmp_path,
        num_epochs=1,
        batch_size=2,
        max_batches_per_epoch=3,
    )
    instance = ModelInstance(name=config.name, config=config, resume_sample_idx=2, resume_epoch=1)
    model = training.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = training._make_train_loader(config, epoch=1, dataset=TinyDataset())

    training.train_epoch(model, loader, optimizer, "cpu", config, instance, epoch=1)

    assert instance.resume_epoch == 2
    assert instance.resume_sample_idx == 0


def test_merge_saved_state_restores_position(tmp_path):
    from model_instance import ModelInstance, TrainingConfig, should_resume_training

    config = TrainingConfig(name="resume_log", run_dir=tmp_path, num_epochs=5)
    instance = ModelInstance(
        name=config.name,
        config=config,
        resume_epoch=2,
        resume_sample_idx=17,
    )
    instance.save_log()

    fresh = ModelInstance(name=config.name, config=config)
    assert fresh.merge_saved_state()
    assert fresh.resume_epoch == 2
    assert fresh.resume_sample_idx == 17
    assert should_resume_training(fresh)


def test_should_resume_false_when_training_complete(tmp_path):
    from model_instance import EpochLog, ModelInstance, TrainingConfig, should_resume_training

    config = TrainingConfig(name="done", run_dir=tmp_path, num_epochs=1)
    instance = ModelInstance(
        name=config.name,
        config=config,
        resume_epoch=2,
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
                gt_bin_recall=0.8,
            )
        ],
    )
    instance.save_log()

    fresh = ModelInstance(name=config.name, config=config)
    assert not should_resume_training(fresh)


def test_train_model_saves_on_keyboard_interrupt(tmp_path, monkeypatch):
    from model_instance import ModelInstance, TrainingConfig

    config = TrainingConfig(
        name="interrupt",
        run_dir=tmp_path,
        num_epochs=3,
        batch_size=2,
        save_every_epochs=5,
        max_batches_per_epoch=1,
    )
    instance = ModelInstance(name=config.name, config=config)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(training, "train_epoch", raise_interrupt)

    with pytest.raises(KeyboardInterrupt):
        training.train_model(instance, resume=False)
    assert instance.pth_path.exists()
    assert instance.log_path.exists()
    assert len(instance.epoch_logs) == 0


def test_checkpoint_timestamp_format():
    from datetime import datetime
    from model_instance import checkpoint_timestamp

    ts = checkpoint_timestamp(datetime(2026, 6, 12, 14, 35))
    assert ts == "12-06_14-35"


def test_checkpoint_log_roundtrip_includes_timestamp(tmp_path):
    from model_instance import CheckpointLog, ModelInstance, TrainingConfig

    config = TrainingConfig(name="ts", run_dir=tmp_path)
    instance = ModelInstance(
        name="ts",
        config=config,
        last_pth_path=tmp_path / "ts" / "ts_12-06_14-35.pth",
        checkpoint_logs=[
            CheckpointLog(
                epoch=1,
                sample_idx=720,
                timestamp="12-06_14-35",
                pth_path=str(tmp_path / "ts" / "ts_12-06_14-35.pth"),
                repeatability=0.1,
                precision=0.2,
                recall=0.3,
                gt_bin_recall=0.4,
                tp=1,
                fp=2,
                fn=3,
                total_gt=4,
                repeatable=0,
                loss_total=1.0,
                loss_keypoint=0.8,
                loss_descriptor=0.2,
            )
        ],
    )
    instance.save_log()
    restored = ModelInstance.load_log(instance.log_path)
    assert restored.checkpoint_logs[0].timestamp == "12-06_14-35"
    assert restored.checkpoint_logs[0].pth_path.endswith("ts_12-06_14-35.pth")
    assert restored.last_pth_path.name == "ts_12-06_14-35.pth"


def test_gt_bin_recall_perfect_match():
    coords = [(40, 32), (80, 80)]
    gt = [torch.tensor([[x, y, 1.0] for x, y in coords], dtype=torch.float32)]
    logits = _logits_peaked_at(coords)
    stats = utils.gt_bin_recall(logits, gt)
    assert stats["recall"] == pytest.approx(1.0, abs=1e-4)
    assert stats["correct"] == 2
    assert stats["total_gt"] == 2


def test_checkpoint_window_triggers_every_n_instances(tmp_path):
    from model_instance import ModelInstance, TrainingConfig

    totals = training._fresh_kpi_totals()
    totals.update({"repeatable": 1, "total_gt": 2, "tp": 3, "fp": 1, "fn": 1})
    gt_bin_totals = {"correct": 4, "total_gt": 10}
    window_running = {"total": 2.0, "keypoint": 1.5, "descriptor": 0.5}

    config = TrainingConfig(name="kpi", run_dir=tmp_path, kpi_every_instances=720)
    instance = ModelInstance(name="kpi", config=config)
    model = training.build_model()

    last_at, new_totals, new_gt_bin, new_running, new_batches = training._maybe_checkpoint_window(
        model, instance, 1, 720, 0, totals, gt_bin_totals, window_running, 2, config,
    )

    assert last_at == 720
    assert new_totals == training._fresh_kpi_totals()
    assert new_gt_bin == training._fresh_gt_bin_totals()
    assert new_running == {}
    assert new_batches == 0
    assert len(instance.checkpoint_logs) == 1
    assert instance.checkpoint_logs[0].sample_idx == 720
    assert instance.checkpoint_logs[0].timestamp
    assert instance.checkpoint_logs[0].pth_path
    assert instance.pth_path.exists()
    assert instance.pth_path.name.startswith("kpi_")
    assert instance.log_path.exists()

    last_at, _, _, _, _ = training._maybe_checkpoint_window(
        model, instance, 1, 500, 0, totals, gt_bin_totals, window_running, 2, config,
    )
    assert last_at == 0


def test_train_epoch_saves_checkpoint_every_n_instances(tmp_path):
    from model_instance import ModelInstance, TrainingConfig

    class TinyDataset:
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            image = torch.zeros(1, conf.CNN_INPUT_HEIGHT, conf.CNN_INPUT_WIDTH)
            return {
                "image_he": image,
                "image_ihc": image,
                "gt_keypoints": torch.tensor([[10.0, 10.0, 1.0]]),
                "meta": {"idx": idx},
            }

    config = TrainingConfig(
        name="ckpt_window",
        run_dir=tmp_path,
        num_epochs=1,
        batch_size=2,
        max_batches_per_epoch=2,
        kpi_every_instances=4,
    )
    instance = ModelInstance(name=config.name, config=config)
    model = training.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = training._make_train_loader(config, epoch=1, dataset=TinyDataset())

    training.train_epoch(model, loader, optimizer, "cpu", config, instance, epoch=1)

    assert len(instance.checkpoint_logs) == 1
    assert instance.checkpoint_logs[0].sample_idx == 4
    assert instance.checkpoint_logs[0].timestamp
    assert Path(instance.checkpoint_logs[0].pth_path).exists()
    assert instance.pth_path.exists()


def test_explore_dataset_render_and_save(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    import explore_dataset

    dataset = StainPairKeypointDataset()
    item = dataset[0]
    arr = explore_dataset.tensor_to_gray_numpy(item["image_he"])
    assert arr.shape == (CNN_H, CNN_W)
    assert arr.min() >= 0.0 and arr.max() <= 1.0

    out_path = tmp_path / "pair.png"
    explore_dataset.save_training_pair(item, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    fig = explore_dataset.render_training_grid(dataset, indices=[0, 1], ncols=2)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
