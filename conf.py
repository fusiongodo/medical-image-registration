from pathlib import Path



CNN_INPUT_HEIGHT = 512
CNN_INPUT_WIDTH = 344

WSI_PAGES = [0,2,3,4]
MAX_CROP_DEPTH = 5

PROJECT_ROOT = Path.cwd().parent
IMAGE_DIR = Path(PROJECT_ROOT, "dataset/images")
LABELS_PATH = Path(PROJECT_ROOT, "setup/labels.json")
ANNOTATION_PATH = Path(PROJECT_ROOT, "dataset/quadtree_annotations.json")