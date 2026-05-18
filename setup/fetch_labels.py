import os
import getpass
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import conf

from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.image_registration_api import ImageRegistrationApi


def main():
    print(f"Detected system: {conf.SYSTEM_PREFIX!r}")
    print(f"Image directory : {conf.IMAGE_DIR}")
    print(f"Output labels   : {conf.LABELS_PATH}")
    print()

    if not conf.IMAGE_DIR.exists():
        print(f"[ERROR] Image directory not found: {conf.IMAGE_DIR}")
        sys.exit(1)

    available_ids: set[int] = {
        int(p.stem)
        for p in conf.IMAGE_DIR.glob("*.data")
        if p.stem.isdigit()
    }

    if not available_ids:
        print(f"[ERROR] No *.data files found in {conf.IMAGE_DIR}")
        sys.exit(1)

    print(f"Found {len(available_ids)} local image(s) in {conf.IMAGE_DIR}")

    configuration = Configuration()
    configuration.username = "alha7503"
    configuration.password = (
        os.getenv("EXACT_PASSWORD") or getpass.getpass("EXACT password: ")
    )
    configuration.host = "https://exact.hs-flensburg.de"

    client = ApiClient(configuration)
    image_registration_api = ImageRegistrationApi(client)

    print("Fetching registration metadata from server...")
    response = image_registration_api.list_image_registrations_with_http_info(limit=4000)
    registrations = response[0].results
    print(f"Server returned {len(registrations)} registration pair(s).")

    labels = []
    for reg in registrations:
        src_id = reg.source_image
        tgt_id = reg.target_image
        if src_id in available_ids and tgt_id in available_ids:
            labels.append({
                "source_image_id": src_id,
                "target_image_id": tgt_id,
                "registration_error": reg.registration_error,
                "transformation_matrix": reg.transformation_matrix,
            })

    print(f"Matched {len(labels)} pair(s) with local images.")

    if not labels:
        print("[WARNING] No matching pairs found – labels file will not be written.")
        return

    conf.LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(conf.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=4)

    print(f"Saved labels to {conf.LABELS_PATH}")


if __name__ == "__main__":
    main()
