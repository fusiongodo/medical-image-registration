import numpy as np


def _as_xy(pts):
    if pts is None or len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts)[:, :2]


def _greedy_match(dist, radius):
    n_a, n_b = dist.shape
    if n_a == 0 or n_b == 0:
        return 0
    flat_order = np.argsort(dist, axis=None)
    matched_a = np.zeros(n_a, dtype=bool)
    matched_b = np.zeros(n_b, dtype=bool)
    n_matched = 0
    for idx in flat_order:
        i, j = divmod(int(idx), n_b)
        if dist[i, j] > radius:
            break
        if matched_a[i] or matched_b[j]:
            continue
        matched_a[i] = True
        matched_b[j] = True
        n_matched += 1
    return n_matched


def match_keypoints_cross_stain(pts_he, pts_ihc, radius: float) -> dict:
    he = _as_xy(pts_he)
    ihc = _as_xy(pts_ihc)
    n_he, n_ihc = len(he), len(ihc)
    if n_he == 0 or n_ihc == 0:
        return {
            "n_he": n_he,
            "n_ihc": n_ihc,
            "n_matched": 0,
            "repeatability": 0.0,
        }
    dist = np.linalg.norm(he[:, None, :] - ihc[None, :, :], axis=2)
    n_matched = _greedy_match(dist, radius)
    return {
        "n_he": n_he,
        "n_ihc": n_ihc,
        "n_matched": n_matched,
        "repeatability": n_matched / max(n_he, n_ihc),
    }


def match_keypoints_to_gt(pts, gt, radius: float) -> int:
    pred = _as_xy(pts)
    gt_xy = _as_xy(gt)
    if len(pred) == 0 or len(gt_xy) == 0:
        return 0
    dist = np.linalg.norm(pred[:, None, :] - gt_xy[None, :, :], axis=2)
    return _greedy_match(dist, radius)
