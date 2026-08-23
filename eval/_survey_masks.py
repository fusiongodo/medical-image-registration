"""Which muROMI pairs have a tissue mask, and how many tiles each yields.

Without a mask, tissue_tiles() returns the full grid including background, so
masked pairs are the ones safe to train on.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "setup" / "live_crop")]

from setup import datasets as ds

ds.set_active_dataset("muromi")
import crop_core
from setup.coarse_to_fine import sp_rot_train_data as data
from setup.labelme import pair_mask

n_pairs = ds.pair_count()
print(f"muromi pair_count={n_pairs}")
print(f"{'pair':>5} {'mask':>6} {'d2':>5} {'d3':>5} {'d4':>5} {'d5':>6}")
masked, unmasked = [], []
totals = {2: 0, 3: 0, 4: 0, 5: 0}
for pid in range(n_pairs):
    try:
        has = pair_mask.load_pair_mask(pid) is not None
    except Exception:
        has = False
    counts = {}
    for d in (2, 3, 4, 5):
        try:
            counts[d] = len(data.scan_tiles([pid], d))
        except Exception:
            counts[d] = -1
    (masked if has else unmasked).append(pid)
    if has:
        for d in (2, 3, 4, 5):
            if counts[d] > 0:
                totals[d] += counts[d]
    print(
        f"{pid:>5} {('yes' if has else 'NO'):>6} "
        + " ".join(f"{counts[d]:>5}" for d in (2, 3, 4))
        + f" {counts[5]:>6}"
    )

print(f"\nmasked pairs ({len(masked)}): {masked}")
print(f"unmasked pairs ({len(unmasked)}): {unmasked}")
print("\ntile totals over masked pairs (x2 for he+ihc):")
for d in (2, 3, 4, 5):
    print(f"  depth {d}: {totals[d]:>6} tiles -> {2 * totals[d]:>6} images")
