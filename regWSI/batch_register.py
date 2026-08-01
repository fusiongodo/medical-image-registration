"""
Serial regWSI registration for many pairs (VPS / GPU). Appends progress.log.

Usage (after unpacking regwsi_inputs_1x.tar.zst at repo root):
  python regWSI/batch_register.py
  python regWSI/batch_register.py --pairs 0
  python regWSI/batch_register.py --smoke          # pair 0 only; require cuda
  python regWSI/batch_register.py --log /tmp/regwsi_progress.log

tail -f data/regwsi/progress.log
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "live_crop"))

import crop_core
import torch

from regWSI import paths
from regWSI.register import register_pair


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(log_path: Path, event: str, **fields) -> None:
    payload = {"ts": _ts(), "event": event, **fields}
    line = json.dumps(payload, ensure_ascii=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


def _expected_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", type=int, help="pair ids (default: all)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="register pair 0 only and fail unless meta.device starts with cuda",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=paths.REGWSI_ROOT / "progress.log",
        help="append-only JSONL progress file",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip pairs that already have displacement_field.mha",
    )
    args = ap.parse_args()

    n = crop_core.num_pairs()
    if args.smoke:
        pair_ids = [0]
    elif args.pairs is not None:
        pair_ids = args.pairs
    else:
        pair_ids = list(range(n))

    for p in pair_ids:
        if p < 0 or p >= n:
            raise SystemExit(f"pair {p} out of range [0, {n})")

    excluded = sorted(p for p in pair_ids if p in paths.EXCLUDED_PAIR_IDS)
    if excluded and not args.smoke:
        pair_ids = [p for p in pair_ids if p not in paths.EXCLUDED_PAIR_IDS]

    log_path = args.log if args.log.is_absolute() else (REPO_ROOT / args.log)
    device = _expected_device()
    _log(
        log_path,
        "BATCH_START",
        n=len(pair_ids),
        pairs=pair_ids,
        excluded_pairs=excluded,
        device_available=device,
        cuda=torch.cuda.is_available(),
        smoke=bool(args.smoke),
    )

    if args.smoke and not torch.cuda.is_available():
        _log(log_path, "FAIL", pair_id=0, error="smoke requires CUDA; torch.cuda.is_available() is False")
        raise SystemExit("smoke failed: no CUDA")

    ok = 0
    failed = 0
    for pair_id in pair_ids:
        if args.skip_existing and paths.displacement_field(pair_id).is_file():
            _log(log_path, "SKIP", pair_id=pair_id, reason="displacement_field exists")
            continue

        he = paths.he_tiff(pair_id)
        ihc = paths.ihc_tiff(pair_id)
        if not he.is_file() or not ihc.is_file():
            _log(log_path, "FAIL", pair_id=pair_id, error="missing he.tiff or ihc.tiff")
            failed += 1
            continue

        _log(log_path, "START", pair_id=pair_id, device=device)
        try:
            meta = register_pair(pair_id, do_export=False, force_export=False)
            used = meta.get("device")
            _log(
                log_path,
                "DEVICE",
                pair_id=pair_id,
                device=used,
                canvas=meta.get("canvas"),
            )
            if args.smoke and not (isinstance(used, str) and used.startswith("cuda")):
                raise RuntimeError(f"smoke expected cuda device, got {used!r}")
            _log(
                log_path,
                "DONE",
                pair_id=pair_id,
                device=used,
                displacement_field=meta.get("displacement_field"),
                warped_ihc=meta.get("warped_ihc"),
            )
            ok += 1
        except Exception as e:
            _log(
                log_path,
                "FAIL",
                pair_id=pair_id,
                error=str(e),
                traceback=traceback.format_exc()[-2000:],
            )
            failed += 1
            if args.smoke:
                _log(log_path, "BATCH_END", ok=ok, failed=failed)
                raise SystemExit(f"smoke failed: {e}") from e

    _log(log_path, "BATCH_END", ok=ok, failed=failed)
    if failed:
        raise SystemExit(f"batch finished with {failed} failure(s), {ok} ok")


if __name__ == "__main__":
    main()
