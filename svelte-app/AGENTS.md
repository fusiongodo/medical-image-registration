# Svelte App — Agent Context

## Image Serving — live crops from raw TIFFs
Tiles are cropped **on the fly from the raw WSI TIFFs**; there is no reliance on
pre-cropped PNGs. A persistent Python worker keeps decoded pyramid pages warm.

- **`$lib/liveCropWorker.ts`** — singleton managing `setup/live_crop/crop_worker.py`
  over stdin/stdout (JSON request/response by id).
- **`api/live-crop/tiles`** — `?pair&level` → tissue tile list (mask-filtered) via `crop_core.tissue_tiles`.
- **`api/live-crop/tile`** — `?pair&level&x&y&side&dx&dy` → PNG. `dx/dy` (optional) are
  **tile-pixel** displacements (512×344 CNN space); the moving IHC is recropped at that
  offset for near-full intersection instead of being canvas-shifted.
- **`api/c2f/fft-map`** — `?pair&depth&tile&dx&dy[&mx&my]` → JSON
  `{image (data-URL PNG), w, h, cx, cy, peaks:[{dx,dy,psr,px,py}], chosen}` for
  the FFT phase-correlation surface at the prior base (worker `fft-map` op,
  `fft_map.py` → `align.surface_and_peaks`). The PNG is the bare colormapped
  surface rendered once at full tile resolution; `FftMap.svelte` is a pan/zoom
  viewer (scroll=zoom, drag=pan, dbl-click=reset) that overlays **fixed-size,
  toggle-able** SVG markers so they never obscure the maxima: grey plus = zero
  shift (`cx,cy`), rank-labelled circles = NMS top-N peaks (rank 0 red = global
  argmax), blue circle = `chosen` (residual behind the current refinement
  vector, `ux - prior_dx`). URL built via `fftMapUrl()`; shown for the
  hovered/selected tile in the C2F tile row.
- **`api/tiles/[pair]/[depth]`** and the page `+page.server.ts` discover tiles via the
  same worker `tiles` op and return `[{ tile }]` (URLs are built client-side).

### Per-tile URLs (built in `[pair]/[depth]/+page.svelte`)
- `heSrc`  = `he`, `dx=dy=0`
- `ihcBaseSrc` = `ihc` at **base offset** → IHC-norm column + `OverlayCanvas`
- `ihcAutoSrc` = `ihc` at **auto offset** → `DisplacedOverlay` (canvas shift is `0`)

Two offsets per tile:
- **base(N)** = previous level's saved field (`/api/field`); **0 at L3**.
- **auto(N)** = per-tile FFT `u,v` from `c2f_cache`, overridden by human `correct`/point-pair.
  Landmark point-pairs are clicked on the base-recropped IHC, so `correct`/manual add the base.

## Persistence — fields only
The only persisted artifacts are the per-level displacement fields; there are **no
per-tile sidecar files and no cropped PNGs** in the data path.

| Endpoint | Reads From | Returns | Used By |
|----------|-----------|---------|---------|
| `api/field` | `smooth_c2f/{pair}_smooth_field.json` `depths[depth]` | `{ [tile]: {dx,dy} }` | base recrop (L4/5) |
| `api/c2f/candidates` | `data/c2f_cache/{pair}_d{depth}.json` | `{cached, candidates:[{tile_loc,u,v,psr,delta_px,by_patch}]}` | auto arrows, LNCC columns, refine approve |
| `api/lncc-distribution` | `c2f_cache` candidates `by_patch[patch].lncc2_auto` | histogram bins/counts | LNCC² distribution panel |
| `api/c2f/refit` | cache + `registration_annotations.json` | per-tile kept/rejected/excluded | C2fHeatmap, C2fVectorField |
| `api/c2f/annotate` | — | writes `registration_annotations.json` | Approve / Correct / Exclude |
| `api/c2f/save-field` | — | shells `refit_cli.py --save` | Writes `smooth_c2f/` |

`c2f_cache` carries the raw per-tile FFT field **and** its LNCC²/Δ metrics, so the grid
columns and histogram derive from it (no separate `metrics.json`). `tileMetrics` and
`autoDisps` in the page are `$derived` from the candidates map.

## Legacy / now-unused (kept on disk, safe to delete later)
`api/image`, `api/python-displacement`, `api/tile-metrics`, `api/lncc-distribution/compute|progress`.
`api/tile-keypoints` still reads `data/cropped/.../keypoints.json` (the master
`macos_he_keypoint_annotations_superpoint.json` is malformed), so `data/cropped/` is the
one remaining dependency blocking its deletion.

## C2F Panel Data Model
`C2fPanel.svelte` receives per-tile results from `api/c2f/refit`:
```ts
interface TileResult {
  tile_loc: string;
  psr: number;
  residual: number;
  kept: boolean;
  excluded?: boolean;
  annotated?: 'approve' | 'correct' | 'exclude' | null;
  dx: number; dy: number;        // fitted field displacement
  ux: number; uy: number;        // FFT candidate (refinement)
  prior_dx: number; prior_dy: number;  // coarser-level field
}
```
- `kept` → green fill, `rejected` → red, `excluded` → grey.
- `annotated` drives stroke outline (approve=yellow, correct=indigo, exclude=grey).

## Alignment
Alignment is triggered from the Coarse-to-fine panel's "compute candidates"
(`api/c2f/candidates` → `run.py --cache-depth <depth>`), which writes `c2f_cache`
(field + metrics); there is no separate displacement-copy or metrics step.

## Keyboard Shortcuts (C2F Panel)
- `Shift+A` — Approve hovered/selected tile
- `Shift+X` — Exclude hovered/selected tile
- `Shift+S` — Clear vote on hovered/selected tile
- `Shift+Q` / `Shift+W` — Toggle HE / IHC overlay emphasis

## Component Conventions
- Heatmap / vector field grids are `460×460` SVGs.
- Tile preview rows use `OverlayCanvas` (static) and `DisplacedOverlay`. In the main grid,
  `DisplacedOverlay` gets an already-recropped `ihcAutoSrc` and shift `0`; the C2F panel
  preview still canvas-shifts by `dx/dy` for its included/excluded comparison.
- C2F panel preview stats are computed in-browser via `computeLNCC()` in `imageUtils.ts`.
