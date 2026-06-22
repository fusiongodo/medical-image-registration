import json
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.path import Path as MplPath
from PIL import Image

import conf

MASK_DIR = conf.PROJECT_ROOT / "data" / "masks"
LABELME_DIR = MASK_DIR / "labelme"

PRODUCTION_MIN_INSIDE_FRACTION = 0.33


def mask_json_path(pair_id):
    return MASK_DIR / f"{pair_id}_he.json"


def mask_preview_path(pair_id):
    return MASK_DIR / f"{pair_id}_he_preview.png"


def labelme_image_path(pair_id):
    return LABELME_DIR / f"{pair_id}_he.png"


def labelme_meta_path(pair_id):
    return LABELME_DIR / f"{pair_id}_he.meta.json"


def labelme_annotation_path(pair_id):
    return LABELME_DIR / f"{pair_id}_he.json"


def load_pair_target(pair_id):
    with open(conf.LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    item = labels[pair_id]
    target_id = item["target_image_id"]
    path = conf.resolve(conf.image_relpath(target_id))
    if not path.exists():
        raise FileNotFoundError(path)
    return target_id, path


def load_he_page_gray(path, pyramid_page_idx):
    with tifffile.TiffFile(path) as slide:
        page = slide.pages[pyramid_page_idx].asarray()
    if page.ndim == 3:
        page = np.dot(page[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif page.dtype != np.uint8:
        page = page.astype(np.uint8)
    return page


def downsample_page(page, max_side):
    height, width = page.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return page, 1.0
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = Image.fromarray(page).convert("L")
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(image, dtype=np.uint8), scale


def image_to_page_coords(points, export_scale):
    inv = 1.0 / export_scale
    return [[x * inv, y * inv] for x, y in points]


def scaled_polygon(polygon, src_width, src_height, dst_width, dst_height):
    if src_width == dst_width and src_height == dst_height:
        return polygon
    sx = dst_width / src_width
    sy = dst_height / src_height
    return [[x * sx, y * sy] for x, y in polygon]


def rasterize_polygon(polygon, width, height):
    if len(polygon) < 3:
        return np.zeros((height, width), dtype=bool)
    path = MplPath(polygon)
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.column_stack((xs.ravel(), ys.ravel()))
    return path.contains_points(points).reshape(height, width)


def rasterize_polygons(polygons, width, height):
    mask = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        mask |= rasterize_polygon(polygon, width, height)
    return mask


def mask_polygons(meta):
    if "polygons" in meta:
        return meta["polygons"]
    return [meta["polygon"]]


def tile_bbox(grid, x_idx, y_idx, page_height, page_width):
    tile_w = page_width // grid
    tile_h = page_height // grid
    x0 = x_idx * tile_w
    y0 = y_idx * tile_h
    x1 = page_width if x_idx == grid - 1 else x0 + tile_w
    y1 = page_height if y_idx == grid - 1 else y0 + tile_h
    return x0, y0, x1, y1


def tile_inside_fraction(mask, grid, x_idx, y_idx):
    page_height, page_width = mask.shape
    x0, y0, x1, y1 = tile_bbox(grid, x_idx, y_idx, page_height, page_width)
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def tile_inside_fraction_from_polygons(
    polygons, grid, x_idx, y_idx, page_height, page_width,
):
    x0, y0, x1, y1 = tile_bbox(grid, x_idx, y_idx, page_height, page_width)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    points = np.column_stack((xs.ravel(), ys.ravel()))
    inside = np.zeros(len(points), dtype=bool)
    for polygon in polygons:
        inside |= MplPath(polygon).contains_points(points)
    return float(inside.mean())


def scaled_polygons_for_page(pair_id, page_width, page_height):
    meta = load_pair_mask(pair_id)
    if meta is None:
        return None
    return [
        scaled_polygon(
            polygon,
            meta["page_width"],
            meta["page_height"],
            page_width,
            page_height,
        )
        for polygon in mask_polygons(meta)
    ]


def is_tile_excluded_by_polygons(
    polygons, grid, x_idx, y_idx, page_height, page_width, min_inside_fraction,
):
    fraction = tile_inside_fraction_from_polygons(
        polygons, grid, x_idx, y_idx, page_height, page_width,
    )
    return fraction < min_inside_fraction


def is_tile_excluded_by_mask(mask, grid, x_idx, y_idx, min_inside_fraction):
    return tile_inside_fraction(mask, grid, x_idx, y_idx) < min_inside_fraction


def save_pair_mask(
    pair_id,
    target_image_id,
    pyramid_page_idx,
    page_width,
    page_height,
    polygon=None,
    polygons=None,
    display_polygon=None,
):
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    if polygons is None:
        if polygon is None:
            raise ValueError("polygon or polygons required")
        polygons = [polygon]
    payload = {
        "pair_id": int(pair_id),
        "target_image_id": int(target_image_id),
        "pyramid_page_idx": int(pyramid_page_idx),
        "page_width": int(page_width),
        "page_height": int(page_height),
        "polygon": [[float(x), float(y)] for x, y in polygons[0]],
        "polygons": [
            [[float(x), float(y)] for x, y in poly]
            for poly in polygons
        ],
    }
    if display_polygon is not None:
        payload["display_polygon"] = [[float(x), float(y)] for x, y in display_polygon]
    path = mask_json_path(pair_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_pair_mask(pair_id):
    path = mask_json_path(pair_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mask_for_page(pair_id, page_width, page_height, pyramid_page_idx=None):
    meta = load_pair_mask(pair_id)
    if meta is None:
        return None
    polygons = []
    for polygon in mask_polygons(meta):
        polygons.append(
            scaled_polygon(
                polygon,
                meta["page_width"],
                meta["page_height"],
                page_width,
                page_height,
            )
        )
    return rasterize_polygons(polygons, page_width, page_height)


def export_labelme_image(pair_id, pyramid_page_idx, export_max_side=8192):
    target_id, path = load_pair_target(pair_id)
    page = load_he_page_gray(path, pyramid_page_idx)
    page_h, page_w = page.shape[:2]
    export_image, export_scale = downsample_page(page, export_max_side)
    export_h, export_w = export_image.shape[:2]

    LABELME_DIR.mkdir(parents=True, exist_ok=True)
    image_path = labelme_image_path(pair_id)
    Image.fromarray(export_image).save(image_path)

    meta = {
        "pair_id": int(pair_id),
        "target_image_id": int(target_id),
        "pyramid_page_idx": int(pyramid_page_idx),
        "page_width": int(page_w),
        "page_height": int(page_h),
        "export_width": int(export_w),
        "export_height": int(export_h),
        "export_scale": float(export_scale),
        "image_path": image_path.name,
        "labelme_shape_label": "valid",
    }
    meta_path = labelme_meta_path(pair_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return image_path, meta_path, meta


def polygons_from_labelme(annotation_path, shape_label=None):
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    all_polygons = []
    matched = []
    for shape in payload.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        poly = [[float(x), float(y)] for x, y in points]
        all_polygons.append((shape.get("label", ""), poly))
        if shape_label is None or shape.get("label", "") == shape_label:
            matched.append(poly)

    if matched:
        return matched
    if all_polygons:
        labels = [label for label, _ in all_polygons]
        print(f"[WARNING] No polygon labeled {shape_label!r}; using all polygons: {labels}")
        return [poly for _, poly in all_polygons]

    labels = sorted(
        {
            shape.get("label", "")
            for shape in payload.get("shapes", [])
            if shape.get("shape_type") == "polygon"
        }
    )
    hint = f" Found polygon labels: {labels}" if labels else ""
    raise ValueError(f"No polygon shapes in {annotation_path}.{hint}")


def import_labelme_to_pair_mask(
    pair_id,
    annotation_path=None,
    meta_path=None,
    shape_label=None,
):
    meta_path = Path(meta_path) if meta_path else labelme_meta_path(pair_id)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if annotation_path is None:
        annotation_path = labelme_annotation_path(pair_id)
    else:
        annotation_path = Path(annotation_path)

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    label = shape_label if shape_label is not None else meta.get("labelme_shape_label")
    export_polygons = polygons_from_labelme(annotation_path, shape_label=label)
    page_polygons = [
        image_to_page_coords(poly, meta["export_scale"])
        for poly in export_polygons
    ]

    out_path = save_pair_mask(
        pair_id=meta["pair_id"],
        target_image_id=meta["target_image_id"],
        pyramid_page_idx=meta["pyramid_page_idx"],
        page_width=meta["page_width"],
        page_height=meta["page_height"],
        polygons=page_polygons,
    )

    mask = rasterize_polygons(
        page_polygons,
        meta["page_width"],
        meta["page_height"],
    )
    save_mask_preview(pair_id, mask)
    return out_path


def save_mask_preview(pair_id, mask):
    preview = (mask.astype(np.uint8) * 255)
    Image.fromarray(preview).save(mask_preview_path(pair_id))
