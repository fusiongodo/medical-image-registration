import platform
from pathlib import Path


CNN_INPUT_HEIGHT = 344
CNN_INPUT_WIDTH = 512

WSI_PAGES = [0,2,3,4]
MAX_CROP_DEPTH = 5

SYSTEM_PREFIX = "macos" if platform.system() == "Darwin" else "win"

PROJECT_ROOT = Path(__file__).parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
LABELS_PATH = PROJECT_ROOT / "setup" / f"{SYSTEM_PREFIX}_labels.json"
ANNOTATION_PATH = PROJECT_ROOT / "data" / f"{SYSTEM_PREFIX}_quadtree_annotations.json"
HE_KEYPOINT_ANNOTATION_PATH = PROJECT_ROOT / "data" / f"{SYSTEM_PREFIX}_he_keypoint_annotations_superpoint.json"