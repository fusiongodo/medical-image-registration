import os
import platform
from pathlib import Path


CNN_INPUT_HEIGHT = 344
CNN_INPUT_WIDTH = 512

WSI_PAGES = [0, 2, 3, 4]
MAX_CROP_DEPTH = 5


def _default_system_prefix():
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Windows":
        return "win"
    return "linux"


SYSTEM_PREFIX = os.environ.get("DATA_PREFIX", _default_system_prefix())

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
LABELS_PATH = PROJECT_ROOT / "data" / f"{SYSTEM_PREFIX}_labels.json"
ANNOTATION_PATH = PROJECT_ROOT / "data" / f"{SYSTEM_PREFIX}_quadtree_annotations.json"
HE_KEYPOINT_ANNOTATION_PATH = (
    PROJECT_ROOT / "data" / f"{SYSTEM_PREFIX}_he_keypoint_annotations_superpoint.json"
)


def image_relpath(image_id: int) -> str:
    return f"data/images/{int(image_id)}.data"


def to_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path outside project root: {path}") from exc


def resolve(stored: str | Path) -> Path:
    path = Path(stored)
    if not path.is_absolute():
        return PROJECT_ROOT / path

    root = PROJECT_ROOT.resolve()
    try:
        return root / path.resolve().relative_to(root)
    except ValueError:
        pass

    parts = path.parts
    for anchor in ("data", "introducing_superpoint"):
        if anchor in parts:
            idx = parts.index(anchor)
            return PROJECT_ROOT.joinpath(*parts[idx:])

    raise ValueError(f"absolute path outside project tree: {stored}")


def resolve_image_path(stored: str | None, image_id: int) -> Path:
    if stored:
        candidate = resolve(stored)
        if candidate.exists():
            return candidate
    fallback = resolve(image_relpath(image_id))
    if fallback.exists():
        return fallback
    if stored:
        return resolve(stored)
    return fallback


def job_image_path(job: dict, side: str) -> Path:
    if side == "fixed":
        image_id = job["target_image_id"]
        stored = job.get("fixed_path")
    else:
        image_id = job["source_image_id"]
        stored = job.get("moving_path")
    return resolve_image_path(stored, image_id)
