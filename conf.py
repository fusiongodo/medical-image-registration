from pathlib import Path



CNN_INPUT_HEIGHT = 344
CNN_INPUT_WIDTH = 512

WSI_PAGES = [0,2,3,4]
MAX_CROP_DEPTH = 5

PROJECT_ROOT = Path.cwd().parent
IMAGE_DIR = Path(PROJECT_ROOT, "data/images")
LABELS_PATH = Path(PROJECT_ROOT, "setup/labels.json")
ANNOTATION_PATH = Path(PROJECT_ROOT, "data/quadtree_annotations.json")