import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from setup.pair_mask import (
    export_labelme_image,
    import_labelme_to_pair_mask,
    labelme_annotation_path,
    mask_preview_path,
)

MODE = "annotate"
PAIR_ID = 1
PYRAMID_PAGE_IDX = 4
EXPORT_MAX_SIDE = 8192
LABELME_SHAPE_LABEL = "valid"


def labelme_executable():
    venv_labelme = Path(sys.executable).parent / "labelme"
    if venv_labelme.exists():
        return str(venv_labelme)
    found = shutil.which("labelme")
    if found:
        return found
    raise FileNotFoundError(
        "labelme not found. Install with: pip install labelme"
    )


def launch_labelme(image_path):
    cmd = [
        labelme_executable(),
        "--autosave",
        "--labels", LABELME_SHAPE_LABEL,
        str(image_path),
    ]
    print(f"Launching LabelMe: {' '.join(cmd)}")
    print("Draw polygon(s), label 'valid', close LabelMe when done.")
    print("(autosave writes JSON next to the PNG on each edit)")
    print()
    subprocess.run(cmd, check=True)


def run_annotate():
    image_path, meta_path, meta = export_labelme_image(
        pair_id=PAIR_ID,
        pyramid_page_idx=PYRAMID_PAGE_IDX,
        export_max_side=EXPORT_MAX_SIDE,
    )

    print(f"Pair ID         : {PAIR_ID}")
    print(f"HE target       : {meta['target_image_id']}")
    print(f"Pyramid page    : {PYRAMID_PAGE_IDX}")
    print(f"Page size       : {meta['page_width']} x {meta['page_height']}")
    print(f"Export size     : {meta['export_width']} x {meta['export_height']}")
    print(f"Image           : {image_path}")
    print(f"Meta            : {meta_path}  (export metadata, not your polygon)")
    print(f"Annotation path : {labelme_annotation_path(PAIR_ID)}")
    print()

    launch_labelme(image_path)

    annotation_path = labelme_annotation_path(PAIR_ID)
    if not annotation_path.exists():
        print(f"[ERROR] No annotation at {annotation_path}")
        print("Re-run and finish the polygon in LabelMe before closing.")
        sys.exit(1)

    print(f"Found annotation: {annotation_path}")
    run_import()


def run_export():
    image_path, meta_path, meta = export_labelme_image(
        pair_id=PAIR_ID,
        pyramid_page_idx=PYRAMID_PAGE_IDX,
        export_max_side=EXPORT_MAX_SIDE,
    )
    print(f"Exported image  : {image_path}")
    print(f"Meta            : {meta_path}")
    print(f"Run manually    : {labelme_executable()} {image_path}")


def run_import():
    out_path = import_labelme_to_pair_mask(
        pair_id=PAIR_ID,
        shape_label=LABELME_SHAPE_LABEL,
    )

    print(f"Pair ID         : {PAIR_ID}")
    print(f"LabelMe JSON    : {labelme_annotation_path(PAIR_ID)}")
    print(f"Saved mask      : {out_path}")
    print(f"Preview mask    : {mask_preview_path(PAIR_ID)}")
    print()
    print("Next: re-run quadtree notebook, then preprocess_tiles.py")


def main():
    if MODE == "annotate":
        run_annotate()
    elif MODE == "export":
        run_export()
    elif MODE == "import":
        run_import()
    else:
        raise ValueError(
            f"Unknown MODE: {MODE!r} (use 'annotate', 'export', or 'import')"
        )


if __name__ == "__main__":
    main()
