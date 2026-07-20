"""
Durable, per-pair "mask out" store for coarse-to-fine registration.

A masked tile is excluded from the field fit and from the training dataset, but
is still cropped/computed and stays explorable in the UI (rendered grey). Masks
propagate FORWARD by quadtree index: masking tile (x, y) at level L masks every
descendant tile at any deeper level L' > L, i.e. all (x', y') with
    x' >> (L' - L) == x   and   y' >> (L' - L) == y
(a level-3 mask covers 4 tiles at level 4, 16 at level 5, and so on).

Per-tile overrides are supported: an explicit entry at a deeper level wins over
an inherited state, so after masking a coarse tile you can re-include (unmask)
an individual descendant. Resolution is "nearest explicit ancestor wins": among
the tile itself and all its ancestors, the deepest-level explicit entry decides.

Single JSON file at data/masked_out.json, keyed by pair:

    {"<pair_id>": [ {"level": 3, "tile_loc": "2_1", "state": "mask"},
                    {"level": 5, "tile_loc": "9_5", "state": "unmask"} ]}

field_set_cli snapshots/restores this file per field set (see its module docs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

MASK_FILE = conf.PROJECT_ROOT / "data" / "masked_out.json"


def load(pair_id: int) -> list[dict]:
    """Return the mask entry list for one pair (empty list if none)."""
    if not MASK_FILE.exists():
        return []
    store = json.loads(MASK_FILE.read_text())
    return list(store.get(str(pair_id), []))


def save(pair_id: int, entries: list[dict]) -> None:
    """Overwrite the mask entry list for one pair, preserving other pairs."""
    store = {}
    if MASK_FILE.exists():
        store = json.loads(MASK_FILE.read_text())
    if entries:
        store[str(pair_id)] = entries
    else:
        store.pop(str(pair_id), None)
    MASK_FILE.parent.mkdir(parents=True, exist_ok=True)
    MASK_FILE.write_text(json.dumps(store, separators=(",", ":")))


def _tile_indices(tile_loc: str) -> tuple[int, int]:
    xi, yi = tile_loc.split("_")
    return int(xi), int(yi)


def _resolve(entries: list[dict], level: int, tile_loc: str, include_self: bool = True) -> bool:
    """Effective masked state for one tile: the deepest-level explicit entry
    among the tile and its ancestors decides; no match -> not masked."""
    x, y = _tile_indices(tile_loc)
    best_level = -1
    best_state: str | None = None
    for e in entries:
        el = int(e["level"])
        if el > level:
            continue
        if not include_self and el == level and e["tile_loc"] == tile_loc:
            continue
        ex, ey = _tile_indices(e["tile_loc"])
        shift = level - el
        if (x >> shift) == ex and (y >> shift) == ey and el > best_level:
            best_level = el
            best_state = e.get("state", "mask")
    return best_state == "mask"


def is_masked(entries: list[dict], level: int, tile_loc: str) -> bool:
    return _resolve(entries, level, tile_loc, include_self=True)


def masked_at(entries: list[dict], level: int, tiles: list[str]) -> set[str]:
    """Subset of `tiles` (all at `level`) whose effective state is masked."""
    return {t for t in tiles if is_masked(entries, level, t)}


def set_effective(pair_id: int, level: int, tile_loc: str, want_masked: bool) -> bool:
    """Force a tile's effective state, storing an explicit override only when it
    differs from what it would inherit (otherwise the override is dropped to keep
    the store minimal). Returns the resulting effective state."""
    entries = [
        e for e in load(pair_id)
        if not (int(e["level"]) == level and e["tile_loc"] == tile_loc)
    ]
    inherited = _resolve(entries, level, tile_loc, include_self=True)
    if want_masked != inherited:
        entries.append({
            "level": level,
            "tile_loc": tile_loc,
            "state": "mask" if want_masked else "unmask",
        })
    save(pair_id, entries)
    return want_masked


def clear(pair_id: int, level: int, tile_loc: str) -> bool:
    """Drop the explicit override at (level, tile_loc), reverting to inherited.
    Returns the resulting effective state."""
    entries = [
        e for e in load(pair_id)
        if not (int(e["level"]) == level and e["tile_loc"] == tile_loc)
    ]
    save(pair_id, entries)
    return _resolve(entries, level, tile_loc, include_self=True)
