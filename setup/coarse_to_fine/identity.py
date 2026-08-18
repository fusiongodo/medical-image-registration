"""
Stable pair-identity fingerprints for coarse-to-fine artifacts.

Pairs are addressed by their index in data/macos_labels.json, but that index is
only stable while the labels file keeps its order. Re-running fetch_labels can
reorder/renumber pairs, which would silently re-associate saved artifacts with
different images. To make drift detectable, every artifact is stamped with the
pair's image-id fingerprint on write, and readers can compare it against the
current labels.

pair_fingerprint(pair_id)            -> {"target_image_id", "source_image_id"}
fingerprint_matches(pair_id, stored) -> bool  (True when absent/legacy)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf


def pair_fingerprint(pair_id: int) -> dict:
    from setup import datasets

    if datasets.uses_pair_tiffs():
        return datasets.pair_fingerprint(int(pair_id))
    labels = json.loads(Path(conf.LABELS_PATH).read_text())
    item = labels[pair_id]
    return {
        "target_image_id": item["target_image_id"],
        "source_image_id": item["source_image_id"],
    }


def fingerprint_matches(pair_id: int, stored: dict | None) -> bool:
    """True if the stored fingerprint matches the current labels.

    A missing/legacy fingerprint (artifact written before stamping existed) is
    tolerated so existing data keeps loading unchanged.
    """
    if not stored:
        return True
    fp = pair_fingerprint(pair_id)
    return (
        stored.get("target_image_id") == fp["target_image_id"]
        and stored.get("source_image_id") == fp["source_image_id"]
    )
