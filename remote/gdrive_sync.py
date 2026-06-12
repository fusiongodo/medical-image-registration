"""
Sync training data to Google Drive via rclone.

Large .data files only appear in the Drive UI after each file completes upload.
Progress is logged to stdout and remote/logs/gdrive_sync.log.

Usage:
  python remote/gdrive_sync.py setup
  python remote/gdrive_sync.py upload
  python remote/gdrive_sync.py upload --images-only
  python remote/gdrive_sync.py status
  python remote/gdrive_sync.py download /path/to/data
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf

DATA_DIR = conf.PROJECT_ROOT / "data"
IMAGES_DIR = conf.IMAGE_DIR
LOG_DIR = REPO_ROOT / "remote" / "logs"

GDRIVE_FOLDER_ID = "1XbxNf8ueM7zLmBP0Fj-HNHehwVB1qNXk"
IMAGES_SUBFOLDER = "images"
REMOTE_NAME = "gdrive_medical"

ANNOTATION_FILES = (
    conf.HE_KEYPOINT_ANNOTATION_PATH.name,
    conf.ANNOTATION_PATH.name,
    conf.LABELS_PATH.name,
)

logger = logging.getLogger("gdrive_sync")


def _ensure_rclone():
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone not found. macOS: brew install rclone | "
            "Linux/RunPod: curl https://rclone.org/install.sh | sudo bash"
        )


def _remote_configured():
    result = subprocess.run(
        ["rclone", "listremotes"],
        capture_output=True,
        text=True,
        check=True,
    )
    return f"{REMOTE_NAME}:" in result.stdout


def setup():
    _ensure_rclone()
    if _remote_configured():
        logger.info("Remote '%s' already configured.", REMOTE_NAME)
        return 0

    logger.info("Opening browser for Google Drive authorization...")
    logger.info("Target folder: https://drive.google.com/drive/folders/%s", GDRIVE_FOLDER_ID)
    subprocess.run(
        [
            "rclone", "config", "create", REMOTE_NAME, "drive",
            "scope", "drive",
            "root_folder_id", GDRIVE_FOLDER_ID,
            "config_is_local", "true",
        ],
        check=True,
    )
    logger.info("Setup complete.")
    return 0


def _format_bytes(num):
    if num is None:
        return "?"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num} B"


def _format_duration(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _log_stats(stats):
    transferred = stats.get("bytes")
    total = stats.get("totalBytes")
    percent = stats.get("percent")
    if percent is None and transferred is not None and total:
        percent = 100.0 * transferred / total

    percent_str = f"{percent:.1f}%" if percent is not None else "?"
    logger.info(
        "progress %s | %s / %s | %s/s | eta %s | files %s / %s",
        percent_str,
        _format_bytes(transferred),
        _format_bytes(total),
        _format_bytes(stats.get("speed")),
        _format_duration(stats.get("eta")),
        stats.get("transfers", "?"),
        stats.get("totalTransfers", "?"),
    )

    transferring = stats.get("transferring") or []
    for item in transferring[:4]:
        name = Path(item.get("name", "?")).name
        size = item.get("size")
        item_bytes = item.get("bytes")
        item_pct = (100.0 * item_bytes / size) if size and item_bytes is not None else None
        pct_str = f"{item_pct:.0f}%" if item_pct is not None else "?"
        logger.info(
            "  -> %s %s (%s/s)",
            name,
            pct_str,
            _format_bytes(item.get("speed")),
        )


def _run_rclone(source, dest, transfers=4):
    cmd = [
        "rclone", "copy", str(source), str(dest),
        f"--drive-root-folder-id={GDRIVE_FOLDER_ID}",
        "--transfers", str(transfers),
        "--checkers", "8",
        "--drive-chunk-size", "64M",
        "--use-json-log",
        "--stats", "5s",
    ]
    logger.info("running: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            logger.info(line)
            continue

        stats = event.get("stats")
        if stats:
            _log_stats(stats)
        elif event.get("msg"):
            logger.info(event["msg"])

    return proc.wait()


def _run_rclone_copy(source, dest, transfers=4):
    return _run_rclone(source, dest, transfers=transfers)


def _running_rclone_uploads():
    result = subprocess.run(
        ["pgrep", "-fl", "rclone copy"],
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def upload_annotations():
    logger.info("Uploading annotations...")
    for name in ANNOTATION_FILES:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        code = _run_rclone_copy(path, f"{REMOTE_NAME}:", transfers=2)
        if code != 0:
            raise RuntimeError(f"annotation upload failed for {name} (exit {code})")
        logger.info("uploaded %s", name)
    return 0


def upload_images(transfers=4):
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(IMAGES_DIR)

    running = _running_rclone_uploads()
    if running:
        logger.warning(
            "Another rclone upload is already running (%d process(es)).",
            len(running),
        )
        for line in running:
            logger.warning("  %s", line)
        logger.warning(
            "Stop it first or let it finish. rclone uploads are resumable."
        )
        return 1

    files = sorted(IMAGES_DIR.glob("*.data"))
    total_bytes = sum(path.stat().st_size for path in files)
    logger.info(
        "Uploading %d image files (~%s) to Drive/%s/",
        len(files),
        _format_bytes(total_bytes),
        IMAGES_SUBFOLDER,
    )
    logger.info(
        "Note: each multi-GB file appears in Drive only after it finishes uploading."
    )
    code = _run_rclone_copy(IMAGES_DIR, f"{REMOTE_NAME}:{IMAGES_SUBFOLDER}", transfers=transfers)
    if code != 0:
        raise RuntimeError(f"image upload failed (exit {code})")
    logger.info("image upload complete")
    return 0


def upload(images_only=False, annotations_only=False, transfers=4):
    _ensure_rclone()
    if not _remote_configured():
        raise RuntimeError("Remote not configured. Run: python remote/gdrive_sync.py setup")

    if not annotations_only:
        if not images_only:
            upload_annotations()
        upload_images(transfers=transfers)
    else:
        upload_annotations()
    return 0


def status():
    _ensure_rclone()
    if not _remote_configured():
        logger.info("Remote not configured.")
        return 1

    logger.info("Root folder contents:")
    subprocess.run(
        [
            "rclone", "ls", f"{REMOTE_NAME}:",
            f"--drive-root-folder-id={GDRIVE_FOLDER_ID}",
        ],
        check=False,
    )
    logger.info("images/ contents:")
    subprocess.run(
        [
            "rclone", "ls", f"{REMOTE_NAME}:{IMAGES_SUBFOLDER}",
            f"--drive-root-folder-id={GDRIVE_FOLDER_ID}",
        ],
        check=False,
    )
    return 0


def download(dest_dir):
    _ensure_rclone()
    if not _remote_configured():
        raise RuntimeError("Remote not configured. Run: python remote/gdrive_sync.py setup")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    images_dest = dest / "images"
    images_dest.mkdir(parents=True, exist_ok=True)

    for name in ANNOTATION_FILES:
        logger.info("Downloading %s", name)
        code = _run_rclone(
            f"{REMOTE_NAME}:{name}",
            dest / name,
            transfers=2,
        )
        if code != 0:
            raise RuntimeError(f"download failed for {name}")

    logger.info("Downloading images/")
    code = _run_rclone(
        f"{REMOTE_NAME}:{IMAGES_SUBFOLDER}",
        images_dest,
        transfers=4,
    )
    if code != 0:
        raise RuntimeError("image download failed")
    logger.info("Download complete: %s", dest)
    return 0


def _configure_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "gdrive_sync.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("Logging to %s", log_path)


def main():
    _configure_logging()

    parser = argparse.ArgumentParser(description="Sync training data with Google Drive")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="One-time Google Drive authorization")

    upload_parser = sub.add_parser("upload", help="Upload annotations and images")
    upload_parser.add_argument("--images-only", action="store_true")
    upload_parser.add_argument("--annotations-only", action="store_true")
    upload_parser.add_argument("--transfers", type=int, default=4)

    sub.add_parser("status", help="List remote files")

    download_parser = sub.add_parser("download", help="Download data to a directory")
    download_parser.add_argument("dest_dir", type=Path)

    args = parser.parse_args()

    if args.command == "setup":
        return setup()
    if args.command == "upload":
        return upload(
            images_only=args.images_only,
            annotations_only=args.annotations_only,
            transfers=args.transfers,
        )
    if args.command == "status":
        return status()
    if args.command == "download":
        return download(args.dest_dir)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
