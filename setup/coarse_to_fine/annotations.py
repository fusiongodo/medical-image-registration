"""
Durable, human-first annotation store for coarse-to-fine registration.

A single JSON file at data/registration_annotations.json holds a flat list of
annotation entries per pair:

    {
      "<pair_id>": [
        {"level": 3, "tile_loc": "5_7", "type": "approve",
         "disp": {"u": 4.1, "v": -2.0}, "source": "fft",   "conf": 0.9},
        {"level": 4, "tile_loc": "9_3", "type": "correct",
         "disp": {"u": 7.5, "v":  1.2}, "source": "human", "conf": 1.0}
      ]
    }

Only HUMAN ACTIONS (approve / correct) are persisted here.  Displacements
`disp.u`/`disp.v` are stored in that level's tile-pixel units (u = dx, v = dy,
in the 512x344 CNN space).  Tau-folded FFT soft points are transient fit inputs
and are never written to this file.

Conversion to fit anchors mirrors setup/smooth_field.py normalisation:
    pos       = ((x + 0.5) / grid, (y + 0.5) / grid)
    disp_frac = (u / (grid * CNN_W), v / (grid * CNN_H))
with grid = 2 ** level.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

ANNOTATION_FILE = conf.PROJECT_ROOT / "data" / "registration_annotations.json"
CNN_W = conf.CNN_INPUT_WIDTH
CNN_H = conf.CNN_INPUT_HEIGHT


@dataclass(frozen=True)
class Anchor:
    """A fit anchor in the shared normalised space (position and displacement in [0,1] fractions)."""
    px: float
    py: float
    du: float
    dv: float
    weight: float


def load(pair_id: int) -> list[dict]:
    """Return the annotation entry list for one pair (empty list if none)."""
    if not ANNOTATION_FILE.exists():
        return []
    store = json.loads(ANNOTATION_FILE.read_text())
    return list(store.get(str(pair_id), []))


def save(pair_id: int, entries: list[dict]) -> None:
    """Overwrite the annotation entry list for one pair, preserving other pairs."""
    store = {}
    if ANNOTATION_FILE.exists():
        store = json.loads(ANNOTATION_FILE.read_text())
    store[str(pair_id)] = entries
    ANNOTATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_FILE.write_text(json.dumps(store, separators=(",", ":")))


def add(pair_id: int, entry: dict) -> list[dict]:
    """Append one annotation entry for a pair and persist. Returns the updated list."""
    entries = load(pair_id)
    entries.append(entry)
    save(pair_id, entries)
    return entries


def remove(pair_id: int, level: int, tile_loc: str) -> list[dict]:
    """Drop all entries for one pair matching (level, tile_loc) and persist. Returns the updated list."""
    entries = [
        e for e in load(pair_id)
        if not (int(e["level"]) == level and e["tile_loc"] == tile_loc)
    ]
    save(pair_id, entries)
    return entries


def make_entry(
    level: int,
    tile_loc: str,
    disp_u: float,
    disp_v: float,
    entry_type: str,
    source: str,
    conf_val: float,
) -> dict:
    return {
        "level": level,
        "tile_loc": tile_loc,
        "type": entry_type,
        "disp": {"u": float(disp_u), "v": float(disp_v)},
        "source": source,
        "conf": float(conf_val),
    }


def _tile_indices(tile_loc: str) -> tuple[int, int]:
    xi, yi = tile_loc.split("_")
    return int(xi), int(yi)


def anchor_from_entry(entry: dict) -> Anchor:
    """Convert one annotation entry to a normalised-space Anchor."""
    level = int(entry["level"])
    grid = 2 ** level
    xi, yi = _tile_indices(entry["tile_loc"])
    u = float(entry["disp"]["u"])
    v = float(entry["disp"]["v"])
    return Anchor(
        px=(xi + 0.5) / grid,
        py=(yi + 0.5) / grid,
        du=u / (grid * CNN_W),
        dv=v / (grid * CNN_H),
        weight=float(entry.get("conf", 1.0)),
    )


def to_anchors(entries: list[dict], up_to_level: int) -> list[Anchor]:
    """All non-excluded entries with level <= up_to_level, converted to normalised anchors."""
    return [
        anchor_from_entry(e)
        for e in entries
        if int(e["level"]) <= up_to_level and e.get("type") != "exclude"
    ]
