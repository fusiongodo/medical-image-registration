"""
Field-set manager for coarse-to-fine registration.

A "field set" is a named, restorable snapshot of one image pair's registration
work: its human annotations (all levels), the fitted smooth field, and the FFT
candidate caches.  Sets live under data/field_sets/{pair}/{set_id}/ and are
plain snapshot / restore copies of the global active-workspace files:

    data/registration_annotations.json      (this pair's slice only)
    data/smooth_c2f/{pair}_smooth_field.json
    data/c2f_cache/{pair}_d{level}.json

The registration tools (align.py / run.py / refit_cli.py / annotate_cli.py) are
untouched: they keep reading the global active files.  This manager only copies
those files in and out of named set folders and tracks which set is active.

Usage (all commands print a JSON result to stdout):
    field_set_cli.py list   <pair>
    field_set_cli.py save   <pair> --name <name> [--id <set_id>]
    field_set_cli.py load   <pair> --id <set_id>
    field_set_cli.py new    <pair> --name <name>
    field_set_cli.py delete <pair> --id <set_id>
    field_set_cli.py rename <pair> --id <set_id> --name <name>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

from setup.coarse_to_fine import annotations

DATA_ROOT = conf.PROJECT_ROOT / "data"
SETS_ROOT = DATA_ROOT / "field_sets"
SMOOTH_DIR = DATA_ROOT / "smooth_c2f"
CACHE_DIR = DATA_ROOT / "c2f_cache"
MAX_DEPTH = conf.MAX_CROP_DEPTH


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(name: str) -> str:
    """Filesystem-safe folder id derived from a set name (case preserved)."""
    slug = _UNSAFE.sub("_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_.-")
    return slug or "set"


def _pair_dir(pair_id: int) -> Path:
    return SETS_ROOT / str(pair_id)


def _set_dir(pair_id: int, set_id: str) -> Path:
    return _pair_dir(pair_id) / set_id


def _active_file(pair_id: int) -> Path:
    return _pair_dir(pair_id) / "active.json"


def _read_active(pair_id: int) -> str | None:
    path = _active_file(pair_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("set_id")
    except Exception:
        return None


def _write_active(pair_id: int, set_id: str | None) -> None:
    path = _active_file(pair_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"set_id": set_id}, separators=(",", ":")))


def _field_path(pair_id: int) -> Path:
    return SMOOTH_DIR / f"{pair_id}_smooth_field.json"


def _cache_paths(pair_id: int) -> list[Path]:
    return sorted(CACHE_DIR.glob(f"{pair_id}_d*.json"))


def _field_meta(pair_id: int) -> dict:
    path = _field_path(pair_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {
        "saved_depth": data.get("saved_depth"),
        "tau": data.get("tau"),
        "n_human": data.get("n_human"),
        "n_kept": data.get("n_kept"),
        "n_seen": data.get("n_seen"),
    }


def _read_manifest(pair_id: int, set_id: str) -> dict | None:
    path = _set_dir(pair_id, set_id) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_sets(pair_id: int) -> dict:
    active = _read_active(pair_id)
    pair_dir = _pair_dir(pair_id)
    sets: list[dict] = []
    if pair_dir.exists():
        for child in sorted(pair_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest = _read_manifest(pair_id, child.name)
            if manifest is not None:
                sets.append(manifest)
    sets.sort(key=lambda m: m.get("updated", 0), reverse=True)
    return {"pair_id": pair_id, "active": active, "sets": sets}


def _snapshot_into(pair_id: int, set_dir: Path) -> None:
    set_dir.mkdir(parents=True, exist_ok=True)

    entries = annotations.load(pair_id)
    (set_dir / "annotations.json").write_text(json.dumps(entries, separators=(",", ":")))

    field_src = _field_path(pair_id)
    field_dst = set_dir / "field.json"
    if field_src.exists():
        shutil.copy2(field_src, field_dst)
    elif field_dst.exists():
        field_dst.unlink()

    cache_dst = set_dir / "candidates"
    if cache_dst.exists():
        shutil.rmtree(cache_dst)
    cache_srcs = _cache_paths(pair_id)
    if cache_srcs:
        cache_dst.mkdir(parents=True, exist_ok=True)
        for src in cache_srcs:
            shutil.copy2(src, cache_dst / src.name)


def save_set(pair_id: int, name: str, set_id: str | None = None) -> dict:
    now = int(time.time())
    slug = _slugify(name)
    existing = _read_manifest(pair_id, slug)
    created = existing.get("created", now) if existing else now

    set_dir = _set_dir(pair_id, slug)
    _snapshot_into(pair_id, set_dir)

    manifest = {
        "id": slug,
        "name": name,
        "pair_id": pair_id,
        "created": created,
        "updated": now,
        **_field_meta(pair_id),
    }
    (set_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    _write_active(pair_id, slug)
    return {"ok": True, "set": manifest}


def load_set(pair_id: int, set_id: str) -> dict:
    set_dir = _set_dir(pair_id, set_id)
    manifest = _read_manifest(pair_id, set_id)
    if manifest is None:
        return {"error": f"no field set {set_id} for pair {pair_id}"}

    ann_path = set_dir / "annotations.json"
    entries = json.loads(ann_path.read_text()) if ann_path.exists() else []
    annotations.save(pair_id, entries)

    field_src = set_dir / "field.json"
    field_dst = _field_path(pair_id)
    if field_src.exists():
        field_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(field_src, field_dst)
    elif field_dst.exists():
        field_dst.unlink()

    for stale in _cache_paths(pair_id):
        stale.unlink()
    cache_src = set_dir / "candidates"
    if cache_src.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for src in sorted(cache_src.glob(f"{pair_id}_d*.json")):
            shutil.copy2(src, CACHE_DIR / src.name)

    _write_active(pair_id, set_id)
    return {"ok": True, "set": manifest}


def new_set(pair_id: int, name: str) -> dict:
    annotations.save(pair_id, [])
    field_dst = _field_path(pair_id)
    if field_dst.exists():
        field_dst.unlink()
    for stale in _cache_paths(pair_id):
        stale.unlink()
    return save_set(pair_id, name)


def delete_set(pair_id: int, set_id: str) -> dict:
    set_dir = _set_dir(pair_id, set_id)
    if set_dir.exists():
        shutil.rmtree(set_dir)
    if _read_active(pair_id) == set_id:
        _write_active(pair_id, None)
    return {"ok": True, "deleted": set_id}


def rename_set(pair_id: int, set_id: str, name: str) -> dict:
    manifest = _read_manifest(pair_id, set_id)
    if manifest is None:
        return {"error": f"no field set {set_id} for pair {pair_id}"}

    new_slug = _slugify(name)
    if new_slug != set_id:
        new_dir = _set_dir(pair_id, new_slug)
        if new_dir.exists():
            return {"error": f"a field set named '{name}' already exists"}
        _set_dir(pair_id, set_id).rename(new_dir)
        if _read_active(pair_id) == set_id:
            _write_active(pair_id, new_slug)

    manifest["id"] = new_slug
    manifest["name"] = name
    manifest["updated"] = int(time.time())
    (_set_dir(pair_id, new_slug) / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"))
    )
    return {"ok": True, "set": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(prog="field_set_cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("pair", type=int)

    p_save = sub.add_parser("save")
    p_save.add_argument("pair", type=int)
    p_save.add_argument("--name", required=True)
    p_save.add_argument("--id", dest="set_id", default=None)

    p_load = sub.add_parser("load")
    p_load.add_argument("pair", type=int)
    p_load.add_argument("--id", dest="set_id", required=True)

    p_new = sub.add_parser("new")
    p_new.add_argument("pair", type=int)
    p_new.add_argument("--name", required=True)

    p_del = sub.add_parser("delete")
    p_del.add_argument("pair", type=int)
    p_del.add_argument("--id", dest="set_id", required=True)

    p_ren = sub.add_parser("rename")
    p_ren.add_argument("pair", type=int)
    p_ren.add_argument("--id", dest="set_id", required=True)
    p_ren.add_argument("--name", required=True)

    args = parser.parse_args()

    if args.command == "list":
        result = list_sets(args.pair)
    elif args.command == "save":
        result = save_set(args.pair, args.name, args.set_id)
    elif args.command == "load":
        result = load_set(args.pair, args.set_id)
    elif args.command == "new":
        result = new_set(args.pair, args.name)
    elif args.command == "delete":
        result = delete_set(args.pair, args.set_id)
    elif args.command == "rename":
        result = rename_set(args.pair, args.set_id, args.name)
    else:
        result = {"error": f"unknown command {args.command}"}

    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
