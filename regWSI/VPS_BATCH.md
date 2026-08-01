# VPS regWSI batch (1×)

Fair vs FFT: inputs are SCALE=1 zlib RGB TIFFs from `export_slides` / `build_rgb_page` (same L5 pyramid page as live crops). Do not JPEG-reencode registration inputs.

## 1. Pack locally

```bash
# requires SCALE=1 in regWSI/paths.py
python regWSI/pack_inputs.py
# or subset:
python regWSI/pack_inputs.py --pairs 0 1 2
```

Writes `data/regwsi/regwsi_inputs_1x.tar.zst` plus `data/regwsi/pack_manifest_1x.json`.

**Excluded pairs** (corrupt JPEG tiles on the L5 pyramid page): `15`, `19`, `20` — see `paths.EXCLUDED_PAIR_IDS`. They are omitted from pack and from `batch_register.py` by default.

Upload the `.tar.zst` to the VPS yourself (scp/rsync).

## 2. Unpack on VPS

From the repo root (same layout as this project):

```bash
zstd -d -c regwsi_inputs_1x.tar.zst | tar -xf -
```

## 3. Smoke CUDA (required before full batch)

VPS needs a CUDA PyTorch build that sees the 5090.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python regWSI/batch_register.py --smoke
# meta must say cuda:
python -c "import json; print(json.load(open('data/regwsi/0/meta.json'))['device'])"
```

`--smoke` registers **pair 0 only** and **exits non-zero** unless `meta.device` starts with `cuda`.

Progress:

```bash
tail -f data/regwsi/progress.log
```

JSONL events: `BATCH_START`, `START`, `DEVICE`, `DONE` / `FAIL`, `BATCH_END`.

## 4. Full serial batch

One pair at a time (do not parallelize on one GPU):

```bash
python regWSI/batch_register.py
# resume-ish:
python regWSI/batch_register.py --skip-existing
```

## 5. Download

Prefer per pair:

- `data/regwsi/{id}/out/displacement_field.mha`
- `data/regwsi/{id}/meta.json`
- `data/regwsi/{id}/out/warped_ihc.tiff` (optional)

Rebuild explorer assets locally:

```bash
python regWSI/make_preview.py <id>
python regWSI/make_full.py <id>
```
