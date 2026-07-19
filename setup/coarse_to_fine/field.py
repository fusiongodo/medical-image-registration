"""
Annotation-derived smooth translation field for coarse-to-fine registration.

Generalises the fitting core of setup/smooth_field.py (which stays untouched):
  * fit a weighted thin-plate-spline field from a set of normalised-space anchors
  * predict per-depth tile-pixel displacements
  * tau-gate candidate FFT displacements against a field prediction
  * robust (tau-based) fit for the headless case with no human anchors
  * write a smooth_field.py-compatible JSON so preprocess_tiles.py can consume it

Anchor weighting is realised through per-point smoothing: RBFInterpolator's
`smoothing` may be a per-point array, and a larger smoothing lets the spline
deviate more at that point.  We set smoothing_i = base_smoothing / weight_i so
high-confidence (human, conf=1) anchors are honoured tightly and low-confidence
FFT soft points are looser.

Coordinate conventions match setup/smooth_field.py:
    pos       normalised to [0,1]^2:  ((x+0.5)/grid, (y+0.5)/grid)
    disp      normalised to image fraction:  u/(grid*CNN_W), v/(grid*CNN_H)
    per-depth output denormalised to tile-pixel (512x344) units.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
from numpy.linalg import LinAlgError
from scipy.interpolate import RBFInterpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import conf

from setup.coarse_to_fine.annotations import Anchor
from setup.coarse_to_fine.identity import pair_fingerprint

CNN_W = conf.CNN_INPUT_WIDTH
CNN_H = conf.CNN_INPUT_HEIGHT
MAX_DEPTH = conf.MAX_CROP_DEPTH
SMOOTH_C2F_DIR = conf.PROJECT_ROOT / "data" / "smooth_c2f"

BASE_SMOOTHING = 1e-3
PSR_CONF_K = 5.0


@dataclass(frozen=True)
class Candidate:
    """A candidate FFT displacement for a single tile, in that level's tile-pixel units."""
    level: int
    tile_loc: str
    u: float
    v: float
    psr: float

    @property
    def conf(self) -> float:
        return psr_to_conf(self.psr)

    def anchor(self) -> Anchor:
        grid = 2 ** self.level
        xi, yi = (int(p) for p in self.tile_loc.split("_"))
        return Anchor(
            px=(xi + 0.5) / grid,
            py=(yi + 0.5) / grid,
            du=self.u / (grid * CNN_W),
            dv=self.v / (grid * CNN_H),
            weight=self.conf,
        )


def candidate_to_dict(c: Candidate) -> dict:
    return {"tile_loc": c.tile_loc, "u": c.u, "v": c.v, "psr": c.psr}


def candidate_from_dict(level: int, d: dict) -> Candidate:
    return Candidate(
        level=level,
        tile_loc=d["tile_loc"],
        u=float(d["u"]),
        v=float(d["v"]),
        psr=float(d["psr"]),
    )


@dataclass
class Field:
    """A fitted (or degenerate) translation field over normalised [0,1]^2 space."""
    kind: str = "identity"                       # "identity" | "constant" | "affine" | "rbf"
    const: tuple[float, float] = (0.0, 0.0)      # normalised (du, dv) for "constant"
    # per-component coefficients (a0, a1, a2) so that d = a0 + a1*x + a2*y (for "affine")
    affine: tuple[tuple[float, float, float], tuple[float, float, float]] | None = dc_field(default=None)
    rbf_dx: object = dc_field(default=None)
    rbf_dy: object = dc_field(default=None)

    def predict_norm(self, pts) -> np.ndarray:
        """pts: (n,2) normalised positions -> (n,2) normalised (du,dv)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self.kind == "identity":
            return np.zeros((len(pts), 2))
        if self.kind == "constant":
            return np.tile(np.asarray(self.const, dtype=float), (len(pts), 1))
        if self.kind == "affine" and self.affine is not None:
            (au, bu, cu), (av, bv, cv) = self.affine
            x, y = pts[:, 0], pts[:, 1]
            return np.stack([au + bu * x + cu * y, av + bv * x + cv * y], axis=1)
        return np.stack([self.rbf_dx(pts), self.rbf_dy(pts)], axis=1)

    def predict_tile_px(self, depth: int) -> dict[str, dict[str, float]]:
        """Evaluate at every tile centre of grid=2**depth; return tile-pixel dx/dy."""
        g = 2 ** depth
        pts, names = [], []
        for yi in range(g):
            for xi in range(g):
                pts.append([(xi + 0.5) / g, (yi + 0.5) / g])
                names.append(f"{xi}_{yi}")
        disp = self.predict_norm(np.asarray(pts))
        out: dict[str, dict[str, float]] = {}
        for name, (du, dv) in zip(names, disp):
            out[name] = {"dx": float(du * g * CNN_W), "dy": float(dv * g * CNN_H)}
        return out

    def predict_tile_px_at(self, level: int, tile_loc: str) -> tuple[float, float]:
        """Tile-pixel (dx, dy) prediction at one tile centre of the given level."""
        g = 2 ** level
        xi, yi = (int(p) for p in tile_loc.split("_"))
        pt = np.array([[(xi + 0.5) / g, (yi + 0.5) / g]])
        du, dv = self.predict_norm(pt)[0]
        return float(du * g * CNN_W), float(dv * g * CNN_H)


def psr_to_conf(psr: float) -> float:
    """Map a peak-to-sidelobe ratio to a confidence in (0,1): psr / (psr + K)."""
    p = max(0.0, float(psr))
    return p / (p + PSR_CONF_K)


def _fit_affine(pts: np.ndarray, du: np.ndarray, dv: np.ndarray, w: np.ndarray) -> Field:
    """
    Weighted least-squares affine field (d = a0 + a1*x + a2*y per component).
    Used when the anchor points are collinear/coincident, where a thin-plate
    spline is singular. lstsq returns the minimum-norm solution, so this degrades
    gracefully to a constant for coincident points while preserving the
    along-line gradient for collinear ones.
    """
    sw = np.sqrt(w)
    design = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1]]) * sw[:, None]
    cu, *_ = np.linalg.lstsq(design, du * sw, rcond=None)
    cv, *_ = np.linalg.lstsq(design, dv * sw, rcond=None)
    return Field(kind="affine", affine=(tuple(map(float, cu)), tuple(map(float, cv))))


def fit_field(anchors: list[Anchor], base_smoothing: float = BASE_SMOOTHING) -> Field:
    """
    Fit a weighted thin-plate-spline field from normalised-space anchors.
      0 anchors            -> identity field (zero displacement)
      1-2 anchors          -> constant field (weighted-mean displacement)
      >=3 collinear anchors -> affine field (TPS is singular for rank-deficient points)
      >=3 anchors          -> thin-plate-spline RBF (per-point smoothing = base / weight)
    """
    if not anchors:
        return Field(kind="identity")

    pts = np.array([[a.px, a.py] for a in anchors], dtype=float)
    du = np.array([a.du for a in anchors], dtype=float)
    dv = np.array([a.dv for a in anchors], dtype=float)
    w = np.array([max(a.weight, 1e-6) for a in anchors], dtype=float)

    if len(anchors) < 3:
        cu = float(np.average(du, weights=w))
        cv = float(np.average(dv, weights=w))
        return Field(kind="constant", const=(cu, cv))

    # The thin-plate spline needs the [1, x, y] monomial matrix at full rank;
    # collinear/coincident anchors make it singular, so fall back to an affine fit.
    if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        return _fit_affine(pts, du, dv, w)

    smoothing = base_smoothing / w
    try:
        rbf_dx = RBFInterpolator(pts, du, kernel="thin_plate_spline", smoothing=smoothing)
        rbf_dy = RBFInterpolator(pts, dv, kernel="thin_plate_spline", smoothing=smoothing)
    except LinAlgError:
        return _fit_affine(pts, du, dv, w)
    return Field(kind="rbf", rbf_dx=rbf_dx, rbf_dy=rbf_dy)


def residuals(candidates: list[Candidate], field: Field) -> list[float]:
    """
    Normalised deviation of each candidate's FFT displacement from the field
    prediction at that tile centre (comparable to tau).
    """
    if not candidates:
        return []
    anchors = [c.anchor() for c in candidates]
    pts = np.array([[a.px, a.py] for a in anchors], dtype=float)
    preds = field.predict_norm(pts)
    out = []
    for a, pred in zip(anchors, preds):
        out.append(float(np.hypot(a.du - pred[0], a.dv - pred[1])))
    return out


def tau_for_keep(human_anchors: list[Anchor], candidates: list[Candidate], keep: float) -> float:
    """
    Derive tau as the `keep`-quantile (0..1) of the candidate residuals, so that
    roughly a `keep` fraction of the auto candidates fall at or below tau and are
    included in the fit (keep=0.5 -> median residual -> ~50% kept; keep=1.0 ->
    all kept).

    The reference field mirrors fit_gated: the human-only field when human
    anchors exist, otherwise an initial fit over all candidates.  Residuals of
    the candidates against that reference form the distribution the quantile is
    taken over.
    """
    if not candidates:
        return 0.0
    keep = min(1.0, max(0.0, keep))
    reference = (
        fit_field(human_anchors)
        if human_anchors
        else fit_field([c.anchor() for c in candidates])
    )
    resid = residuals(candidates, reference)
    if not resid:
        return 0.0
    return float(np.quantile(np.asarray(resid, dtype=float), keep))


def tau_gate(
    candidates: list[Candidate], field: Field, tau: float
) -> tuple[list[Candidate], list[Candidate]]:
    """
    Split candidates by deviation of their FFT displacement from the field
    prediction (in normalised units): <= tau -> kept (fold in as soft points),
    > tau -> rejected (FFT failed there).
    """
    kept: list[Candidate] = []
    rejected: list[Candidate] = []
    for cand, dev in zip(candidates, residuals(candidates, field)):
        (kept if dev <= tau else rejected).append(cand)
    return kept, rejected


def fit_gated(
    human_anchors: list[Anchor],
    candidates: list[Candidate],
    tau: float,
    base_smoothing: float = BASE_SMOOTHING,
) -> tuple[Field, list[Candidate]]:
    """
    Fit the field for one level, honouring human anchors and tau-gating FFT soft points.
      * human_anchors present -> fit a human-only field, tau-gate candidates against it,
        then refit on the union (human anchors + kept soft points).
      * no human anchors       -> headless robust_fit over the candidates.
    Returns (field, kept_candidates).
    """
    if human_anchors:
        hfield = fit_field(human_anchors, base_smoothing)
        kept, _ = tau_gate(candidates, hfield, tau)
        field = fit_field(human_anchors + [c.anchor() for c in kept], base_smoothing)
        return field, kept
    return robust_fit(candidates, tau, base_smoothing)


def robust_fit(
    candidates: list[Candidate],
    tau: float,
    base_smoothing: float = BASE_SMOOTHING,
    max_iter: int = 5,
) -> tuple[Field, list[Candidate]]:
    """
    Headless bootstrap when no human anchors exist: fit all candidates, drop those
    whose residual to the fit exceeds tau, refit, until the kept set is stable.
    Returns (field, kept_candidates).
    """
    kept = list(candidates)
    field = fit_field([c.anchor() for c in kept], base_smoothing)
    for _ in range(max_iter):
        inliers, _ = tau_gate(kept, field, tau)
        if len(inliers) == len(kept):
            break
        if len(inliers) < 3:
            kept = inliers
            field = fit_field([c.anchor() for c in kept], base_smoothing)
            break
        kept = inliers
        field = fit_field([c.anchor() for c in kept], base_smoothing)
    return field, kept


def write_field_json(
    pair_id: int,
    field: Field,
    max_depth: int = MAX_DEPTH,
    meta: dict | None = None,
    out_dir: Path = SMOOTH_C2F_DIR,
) -> Path:
    """Emit a setup/smooth_field.py-compatible JSON to data/smooth_c2f/{pair}_smooth_field.json."""
    depths_out = {str(d): field.predict_tile_px(d) for d in range(max_depth + 1)}
    result: dict = {
        "pair_id": pair_id,
        "identity": pair_fingerprint(pair_id),
        "fit_depth": max_depth,
        "depths": depths_out,
    }
    if meta:
        result.update(meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair_id}_smooth_field.json"
    out_path.write_text(json.dumps(result, separators=(",", ":")))
    return out_path
