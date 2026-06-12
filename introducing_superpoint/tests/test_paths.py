import importlib
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PKG_ROOT.parent))
sys.path.append(str(PKG_ROOT))

import conf
importlib.reload(conf)


def test_image_relpath():
    assert conf.image_relpath(6045) == "data/images/6045.data"


def test_resolve_relative():
    path = conf.resolve("data/images/6045.data")
    assert path == conf.PROJECT_ROOT / "data" / "images" / "6045.data"


def test_resolve_legacy_absolute():
    legacy = f"{conf.PROJECT_ROOT}/data/images/6045.data"
    assert conf.resolve(legacy) == conf.PROJECT_ROOT / "data" / "images" / "6045.data"


def test_resolve_legacy_absolute_under_introducing_superpoint():
    legacy = (
        f"{conf.PROJECT_ROOT}/introducing_superpoint/"
        "superpoint_v6_from_tf.pth"
    )
    assert conf.resolve(legacy) == (
        conf.PROJECT_ROOT / "introducing_superpoint" / "superpoint_v6_from_tf.pth"
    )


def test_resolve_legacy_absolute_foreign_machine():
    legacy = "/old/home/user/medical-image-registration/introducing_superpoint/runs/smoke/smoke.pth"
    assert conf.resolve(legacy) == (
        conf.PROJECT_ROOT / "introducing_superpoint" / "runs" / "smoke" / "smoke.pth"
    )


def test_to_relative_roundtrip():
    original = conf.PROJECT_ROOT / "introducing_superpoint" / "runs"
    assert conf.resolve(conf.to_relative(original)) == original.resolve()


def test_job_image_path_from_ids():
    job = {
        "fixed_path": "data/images/6045.data",
        "moving_path": "data/images/6036.data",
        "target_image_id": 6045,
        "source_image_id": 6036,
    }
    assert conf.job_image_path(job, "fixed").name == "6045.data"
    assert conf.job_image_path(job, "moving").name == "6036.data"


def test_job_image_path_ignores_bad_legacy_absolute():
    job = {
        "fixed_path": "/old/machine/data/images/6045.data",
        "moving_path": "/old/machine/data/images/6036.data",
        "target_image_id": 6045,
        "source_image_id": 6036,
    }
    fixed = conf.job_image_path(job, "fixed")
    moving = conf.job_image_path(job, "moving")
    assert fixed == conf.resolve(conf.image_relpath(6045))
    assert moving == conf.resolve(conf.image_relpath(6036))
