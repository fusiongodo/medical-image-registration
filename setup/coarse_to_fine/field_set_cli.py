"""
Field-set manager for coarse-to-fine registration.

A "field set" is a named, restorable snapshot of one image pair's registration
work: its human annotations (all levels), the fitted smooth field, and the FFT
candidate caches.  Sets live under
data/curated_field_sets/{lam}/{field_estimator}/{pair}/{set_id}/ and are
plain snapshot / restore copies of the global active-workspace files:

    data/registration_annotations.json      (this pair's slice only)
    data/smooth_c2f/{pair}_smooth_field.json
    data/c2f_cache/{pair}_d{level}.json              (FFT LAM)
    data/c2f_cache/{lam}/{pair}_d{level}.json         (other LAMs)

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

from setup.coarse_to_fine import annotations, deskew, masks, rigid_sp_lg as rigid
from setup.coarse_to_fine.identity import pair_fingerprint
from setup.coarse_to_fine.reg_branches import (
    DEFAULT_FIELD_ESTIMATOR,
    DEFAULT_LAM,
    FIELD_ESTIMATORS,
    LAMS,
    branch_root,
    cache_dir,
    cache_paths,
    normalize_estimator,
    normalize_lam,
)

DATA_ROOT = conf.PROJECT_ROOT / "data"
SMOOTH_DIR = DATA_ROOT / "smooth_c2f"
MAX_DEPTH = conf.MAX_CROP_DEPTH

_BRANCH_LAM = DEFAULT_LAM
_BRANCH_ESTIMATOR = DEFAULT_FIELD_ESTIMATOR

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def configure_branch(lam: str | None = None, field_estimator: str | None = None) -> None:
    global _BRANCH_LAM, _BRANCH_ESTIMATOR
    _BRANCH_LAM = normalize_lam(lam)
    _BRANCH_ESTIMATOR = normalize_estimator(field_estimator)


def _sets_root() -> Path:
    return branch_root(_BRANCH_LAM, _BRANCH_ESTIMATOR)


def _slugify(name: str) -> str:
    """Filesystem-safe folder id derived from a set name (case preserved)."""
    slug = _UNSAFE.sub("_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_.-")
    return slug or "set"


def _pair_dir(pair_id: int) -> Path:
    return _sets_root() / str(pair_id)


def _set_dir(pair_id: int, set_id: str) -> Path:
    return _pair_dir(pair_id) / set_id


def _active_file(pair_id: int) -> Path:
    return _pair_dir(pair_id) / "active.json"


def _read_pair_state(pair_id: int) -> dict:
    path = _active_file(pair_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_pair_state(pair_id: int, **updates) -> None:
    state = _read_pair_state(pair_id)
    state.update(updates)
    path = _active_file(pair_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, separators=(",", ":")))


def _read_active(pair_id: int) -> str | None:
    return _read_pair_state(pair_id).get("set_id")


def _write_active(pair_id: int, set_id: str | None) -> None:
    _write_pair_state(pair_id, set_id=set_id)


def _read_main(pair_id: int) -> str | None:
    return _read_pair_state(pair_id).get("main_set_id")


def _write_main(pair_id: int, set_id: str | None) -> None:
    _write_pair_state(pair_id, main_set_id=set_id)


def _field_path(pair_id: int) -> Path:
    return SMOOTH_DIR / f"{pair_id}_smooth_field.json"


def _cache_paths(pair_id: int) -> list[Path]:
    return cache_paths(pair_id, _BRANCH_LAM, field_estimator=_BRANCH_ESTIMATOR)


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
    main = _read_main(pair_id)
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
    return {
        "pair_id": pair_id,
        "lam": _BRANCH_LAM,
        "field_estimator": _BRANCH_ESTIMATOR,
        "active": active,
        "main": main,
        "sets": sets,
    }


def _snapshot_into(pair_id: int, set_dir: Path) -> None:
    set_dir.mkdir(parents=True, exist_ok=True)

    entries = annotations.load(pair_id)
    (set_dir / "annotations.json").write_text(json.dumps(entries, separators=(",", ":")))

    mask_entries = masks.load(pair_id)
    (set_dir / "masked_out.json").write_text(json.dumps(mask_entries, separators=(",", ":")))

    deskew_store = deskew.load(pair_id)
    deskew_dst = set_dir / "deskew.json"
    if deskew_store is not None:
        deskew_dst.write_text(json.dumps(deskew_store, separators=(",", ":")))
    elif deskew_dst.exists():
        deskew_dst.unlink()

    rigid_store = rigid.load(pair_id)
    rigid_dst = set_dir / "rigid_light_v1.json"
    if rigid_store is not None:
        rigid_dst.write_text(json.dumps(rigid_store, separators=(",", ":")))
    elif rigid_dst.exists():
        rigid_dst.unlink()

    matches_store = rigid.load_matches(pair_id)
    matches_dst = set_dir / "rigid_light_v1.matches.json"
    if matches_store is not None:
        matches_dst.write_text(json.dumps(matches_store, separators=(",", ":")))
    elif matches_dst.exists():
        matches_dst.unlink()

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
        "lam": _BRANCH_LAM,
        "field_estimator": _BRANCH_ESTIMATOR,
        "identity": pair_fingerprint(pair_id),
        "created": created,
        "updated": now,
        "rating": existing.get("rating") if existing else None,
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

    mask_path = set_dir / "masked_out.json"
    mask_entries = json.loads(mask_path.read_text()) if mask_path.exists() else []
    masks.save(pair_id, mask_entries)

    deskew_path = set_dir / "deskew.json"
    deskew.write(pair_id, json.loads(deskew_path.read_text()) if deskew_path.exists() else None)

    rigid_path = set_dir / "rigid_light_v1.json"
    rigid.write(pair_id, json.loads(rigid_path.read_text()) if rigid_path.exists() else None)

    matches_path = set_dir / "rigid_light_v1.matches.json"
    rigid.write_matches(
        pair_id,
        json.loads(matches_path.read_text()) if matches_path.exists() else None,
    )

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
        dst_dir = cache_dir(_BRANCH_LAM, field_estimator=_BRANCH_ESTIMATOR)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(cache_src.glob(f"{pair_id}_d*.json")):
            shutil.copy2(src, dst_dir / src.name)

    _write_active(pair_id, set_id)
    return {"ok": True, "set": manifest}


def new_set(pair_id: int, name: str) -> dict:
    annotations.save(pair_id, [])
    masks.save(pair_id, [])
    deskew.clear(pair_id)
    rigid.clear(pair_id)
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
    if _read_main(pair_id) == set_id:
        _write_main(pair_id, None)
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
        if _read_main(pair_id) == set_id:
            _write_main(pair_id, new_slug)

    manifest["id"] = new_slug
    manifest["name"] = name
    manifest["updated"] = int(time.time())
    (_set_dir(pair_id, new_slug) / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"))
    )
    return {"ok": True, "set": manifest}


_RATINGS = {"bad", "ok", "good"}


def rate_set(pair_id: int, set_id: str, rating: str) -> dict:
    if rating not in _RATINGS:
        return {"error": f"rating must be one of {sorted(_RATINGS)}"}
    manifest = _read_manifest(pair_id, set_id)
    if manifest is None:
        return {"error": f"no field set {set_id} for pair {pair_id}"}
    manifest["rating"] = rating
    manifest["updated"] = int(time.time())
    (_set_dir(pair_id, set_id) / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"))
    )
    return {"ok": True, "set": manifest}


def set_main(pair_id: int, set_id: str) -> dict:
    manifest = _read_manifest(pair_id, set_id)
    if manifest is None:
        return {"error": f"no field set {set_id} for pair {pair_id}"}
    _write_main(pair_id, set_id)
    return {"ok": True, "main": set_id}


def _add_branch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lam", default=DEFAULT_LAM, choices=LAMS)
    p.add_argument(
        "--field-estimator",
        dest="field_estimator",
        default=DEFAULT_FIELD_ESTIMATOR,
        choices=FIELD_ESTIMATORS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="field_set_cli.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("pair", type=int)
    _add_branch_args(p_list)

    p_save = sub.add_parser("save")
    p_save.add_argument("pair", type=int)
    p_save.add_argument("--name", required=True)
    p_save.add_argument("--id", dest="set_id", default=None)
    _add_branch_args(p_save)

    p_load = sub.add_parser("load")
    p_load.add_argument("pair", type=int)
    p_load.add_argument("--id", dest="set_id", required=True)
    _add_branch_args(p_load)

    p_new = sub.add_parser("new")
    p_new.add_argument("pair", type=int)
    p_new.add_argument("--name", required=True)
    _add_branch_args(p_new)

    p_del = sub.add_parser("delete")
    p_del.add_argument("pair", type=int)
    p_del.add_argument("--id", dest="set_id", required=True)
    _add_branch_args(p_del)

    p_ren = sub.add_parser("rename")
    p_ren.add_argument("pair", type=int)
    p_ren.add_argument("--id", dest="set_id", required=True)
    p_ren.add_argument("--name", required=True)
    _add_branch_args(p_ren)

    p_rate = sub.add_parser("rate")
    p_rate.add_argument("pair", type=int)
    p_rate.add_argument("--id", dest="set_id", required=True)
    p_rate.add_argument("--rating", required=True, choices=sorted(_RATINGS))
    _add_branch_args(p_rate)

    p_main = sub.add_parser("main")
    p_main.add_argument("pair", type=int)
    p_main.add_argument("--id", dest="set_id", required=True)
    _add_branch_args(p_main)

    args = parser.parse_args()
    configure_branch(args.lam, args.field_estimator)

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
    elif args.command == "rate":
        result = rate_set(args.pair, args.set_id, args.rating)
    elif args.command == "main":
        result = set_main(args.pair, args.set_id)
    else:
        result = {"error": f"unknown command {args.command}"}

    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
