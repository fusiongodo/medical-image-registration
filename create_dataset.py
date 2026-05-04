import os
import getpass
import json
from pathlib import Path
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.api.image_registration_api import ImageRegistrationApi
from exact_sync.v1.api.images_api import ImagesApi

configuration = Configuration()
configuration.username = 'alha7503'
configuration.password = os.getenv("EXACT_PASSWORD") or getpass.getpass("EXACT password: ")
configuration.host = "https://exact.hs-flensburg.de"

client = ApiClient(configuration)
image_registration_api = ImageRegistrationApi(client)
images_api = ImagesApi(client)

DATASET_DIR = Path("dataset/images")
DATASET_DIR.mkdir(parents=True, exist_ok=True)
LABEL_FILE = Path("dataset/labels.json")

MAX_DATA_BYTES = 50 * 1024 * 1024 * 1024
current_data_bytes = 0
dataset_labels = []

print("Fetching registration metadata from server...")
response = image_registration_api.list_image_registrations_with_http_info(limit=4000)
registrations = response[0].results

print(f"Found {len(registrations)} registration pairs. Starting download pipeline...")

for reg in registrations:
    if current_data_bytes >= MAX_DATA_BYTES:
        print("\n[!] 50GB Limit reached. Halting downloads.")
        break

    src_id = reg.source_image
    tgt_id = reg.target_image
    matrix = reg.transformation_matrix
    error = reg.registration_error

    src_path = DATASET_DIR / f"{src_id}.data"
    tgt_path = DATASET_DIR / f"{tgt_id}.data"

    try:
        if not src_path.exists():
            print(f"Downloading source image {src_id}...")
            images_api.download_image(id=src_id, target_path=str(src_path))
            current_data_bytes += src_path.stat().st_size

        if not tgt_path.exists():
            print(f"Downloading target image {tgt_id}...")
            images_api.download_image(id=tgt_id, target_path=str(tgt_path))
            current_data_bytes += tgt_path.stat().st_size

        dataset_labels.append({
            "source_image_id": src_id,
            "target_image_id": tgt_id,
            "registration_error": error,
            "transformation_matrix": matrix
        })
        
        print(f"Successfully processed pair: {src_id} -> {tgt_id}. Total size so far: {current_data_bytes / (1024**3):.2f} GB")

    except Exception as e:
        print(f"Failed to process pair {src_id} -> {tgt_id}. Error: {e}")
        continue

print(f"\nWriting labels to {LABEL_FILE}...")
with open(LABEL_FILE, "w") as f:
    json.dump(dataset_labels, f, indent=4)

print("Done! Dataset is ready for training.")