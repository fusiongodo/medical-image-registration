"""
Rewrite fixed_path / moving_path in annotation JSON to project-relative paths.

Usage:
  python setup/migrate_annotation_paths.py
  python setup/migrate_annotation_paths.py --path data/macos_he_keypoint_annotations_superpoint.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf


def _normalize_job_paths(job: dict) -> bool:
    changed = False
    if "target_image_id" in job:
        rel = conf.image_relpath(job["target_image_id"])
        if job.get("fixed_path") != rel:
            job["fixed_path"] = rel
            changed = True
    if "source_image_id" in job:
        rel = conf.image_relpath(job["source_image_id"])
        if job.get("moving_path") != rel:
            job["moving_path"] = rel
            changed = True
    return changed


def migrate_file(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"skip {path.name}: expected list")
        return 0

    changed = sum(_normalize_job_paths(job) for job in data)
    if changed == 0:
        print(f"ok {path.name}: already relative ({len(data)} jobs)")
        return 0

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"migrated {path.name}: {changed}/{len(data)} jobs updated")
    return changed


def main():
    parser = argparse.ArgumentParser(description="Migrate annotation JSON paths to relative")
    parser.add_argument("--path", type=Path, action="append")
    args = parser.parse_args()

    paths = args.path or [
        conf.ANNOTATION_PATH,
        conf.HE_KEYPOINT_ANNOTATION_PATH,
    ]

    total = 0
    for path in paths:
        if not path.exists():
            alt = REPO_ROOT / "data" / path.name
            if alt.exists():
                path = alt
            else:
                print(f"missing {path}")
                continue
        total += migrate_file(path)
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
