"""Phase-1 coarse-to-fine foundation tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndimage_shift

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "setup" / "auto-alignment"))

import conf  # noqa: E402
import align  # noqa: E402
from setup.coarse_to_fine.annotations import Anchor, anchor_from_entry, make_entry  # noqa: E402
from setup.coarse_to_fine.field import (  # noqa: E402
    Candidate,
    Field,
    fit_field,
    fit_gated,
    psr_to_conf,
    tau_gate,
)

CNN_W = conf.CNN_INPUT_WIDTH
CNN_H = conf.CNN_INPUT_HEIGHT


# ── 1. Normalisation round-trip ──────────────────────────────────────────────

@pytest.mark.parametrize("level", [3, 4, 5])
def test_normalization_round_trip(level):
    grid = 2 ** level
    u, v = 7.5, -3.25
    tile_loc = f"{grid // 2}_{grid // 3}"
    entry = make_entry(level, tile_loc, u, v, "correct", "human", 1.0)
    a = anchor_from_entry(entry)

    # normalised displacement back to tile-pixel units
    assert a.du * grid * CNN_W == pytest.approx(u)
    assert a.dv * grid * CNN_H == pytest.approx(v)

    # position is the tile centre in [0,1]^2
    xi, yi = grid // 2, grid // 3
    assert a.px == pytest.approx((xi + 0.5) / grid)
    assert a.py == pytest.approx((yi + 0.5) / grid)


# ── 2. Composition: coarse + residual recovers a uniform ground-truth shift ──

def test_uniform_field_prediction_recovers_shift():
    level = 4
    grid = 2 ** level
    gt_dx, gt_dy = 6.0, -4.0
    du = gt_dx / (grid * CNN_W)
    dv = gt_dy / (grid * CNN_H)

    anchors = [
        Anchor(px=(x + 0.5) / grid, py=(y + 0.5) / grid, du=du, dv=dv, weight=1.0)
        for x, y in [(2, 2), (10, 3), (5, 12), (13, 13), (8, 7)]
    ]
    field = fit_field(anchors)

    dx, dy = field.predict_tile_px_at(level, "6_6")
    assert dx == pytest.approx(gt_dx, abs=1e-3)
    assert dy == pytest.approx(gt_dy, abs=1e-3)


def test_composition_total_is_coarse_plus_residual():
    level = 5
    grid = 2 ** level
    coarse_dx, coarse_dy = 5.0, 2.0
    residual_dx, residual_dy = 1.5, -0.5

    # a constant coarse field
    coarse = Field(kind="constant", const=(coarse_dx / (grid * CNN_W), coarse_dy / (grid * CNN_H)))
    cdx, cdy = coarse.predict_tile_px_at(level, "10_10")
    assert cdx == pytest.approx(coarse_dx)
    assert cdy == pytest.approx(coarse_dy)

    total_u = cdx + residual_dx
    total_v = cdy + residual_dy
    assert total_u == pytest.approx(coarse_dx + residual_dx)
    assert total_v == pytest.approx(coarse_dy + residual_dy)


# ── 3. tau gate ──────────────────────────────────────────────────────────────

def test_tau_gate_rejects_outlier():
    level = 4
    field = Field(kind="identity")  # predicts zero displacement everywhere

    grid = 2 ** level
    small = 3.0 / (grid * CNN_W)          # a few px -> small normalised deviation
    big = 80.0 / (grid * CNN_W)           # large deviation
    tau = 10.0 / (grid * CNN_W)

    inlier = Candidate(level=level, tile_loc="4_4", u=3.0, v=0.0, psr=12.0)
    outlier = Candidate(level=level, tile_loc="5_5", u=80.0, v=0.0, psr=2.0)

    kept, rejected = tau_gate([inlier, outlier], field, tau)
    assert inlier in kept
    assert outlier in rejected
    assert small < tau < big


def test_psr_to_conf_monotonic():
    assert psr_to_conf(0.0) == pytest.approx(0.0)
    assert 0.0 < psr_to_conf(5.0) < psr_to_conf(20.0) < 1.0


# ── 3b. fit_gated: human-first refinement ────────────────────────────────────

def test_fit_gated_human_overrides_conflicting_fft():
    level = 4
    grid = 2 ** level
    tau = 5.0 / (grid * CNN_W)  # tight gate in normalised units

    # FFT insists on zero displacement everywhere...
    cands = [
        Candidate(level=level, tile_loc=f"{x}_{y}", u=0.0, v=0.0, psr=10.0)
        for x, y in [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (2, 10), (10, 2)]
    ]
    # ...but a human corrects tile 6_6 to a large, conflicting shift.
    hu, hv = 12.0, -8.0
    human = [anchor_from_entry(make_entry(level, "6_6", hu, hv, "correct", "human", 1.0))]

    field, kept = fit_gated(human, cands, tau)

    dx, dy = field.predict_tile_px_at(level, "6_6")
    assert dx == pytest.approx(hu, abs=1.0)
    assert dy == pytest.approx(hv, abs=1.0)
    # the conflicting FFT soft points are gated out
    assert len(kept) == 0


def test_fit_gated_honours_all_human_anchors():
    level = 4
    specs = {
        "2_2": (3.0, 1.0),
        "12_3": (9.0, -2.0),
        "6_12": (-4.0, 5.0),
        "13_13": (0.5, -0.5),
    }
    human = [
        anchor_from_entry(make_entry(level, loc, u, v, "correct", "human", 1.0))
        for loc, (u, v) in specs.items()
    ]

    field, _ = fit_gated(human, [], tau=1e-3)

    for loc, (u, v) in specs.items():
        dx, dy = field.predict_tile_px_at(level, loc)
        assert dx == pytest.approx(u, abs=0.5)
        assert dy == pytest.approx(v, abs=0.5)


def test_fit_gated_falls_back_to_robust_without_human():
    level = 4
    cands = [
        Candidate(level=level, tile_loc=f"{x}_{y}", u=2.0, v=1.0, psr=10.0)
        for x, y in [(2, 2), (12, 3), (5, 11), (13, 13), (7, 6)]
    ]
    field, kept = fit_gated([], cands, tau=1e-2)
    dx, dy = field.predict_tile_px_at(level, "6_6")
    assert dx == pytest.approx(2.0, abs=1.0)
    assert dy == pytest.approx(1.0, abs=1.0)
    assert len(kept) == len(cands)


# ── 4. register_arrays: shift-sign convention + PSR ──────────────────────────

def _textured_image(seed: int, h: int = 128, w: int = 160) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.standard_normal((h, w)), sigma=3.0)
    img -= img.min()
    img *= 255.0 / (img.max() + 1e-9)
    return img


def test_register_arrays_recovers_shift_sign():
    he = _textured_image(seed=0)
    sx = 6.0  # IHC content displaced to the right
    ihc = ndimage_shift(he, shift=(0.0, sx), order=1, mode="nearest")

    res = align.register_arrays(he, ihc)

    # aligning IHC onto HE requires shifting it back left -> dx must be negative
    assert res["dx"] < 0
    assert res["dx"] == pytest.approx(-sx, abs=2.0)
    assert abs(res["dy"]) < 2.0


def test_register_arrays_psr_higher_for_matched_pair():
    he = _textured_image(seed=1)
    ihc_match = ndimage_shift(he, shift=(0.0, 4.0), order=1, mode="nearest")
    ihc_noise = _textured_image(seed=99)

    psr_match = align.register_arrays(he, ihc_match)["psr"]
    psr_noise = align.register_arrays(he, ihc_noise)["psr"]

    assert psr_match > psr_noise
