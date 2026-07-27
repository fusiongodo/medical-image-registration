/**
 * Single source of truth for the /api/live-crop/tile URL.
 *
 * The moving IHC is recropped from the raw WSI at the given offset (baked into
 * the crop, never CSS-translated), so large per-level displacements still keep
 * full intersection with the fixed HE tile. dx/dy of 0 are omitted so the
 * URL for an unshifted crop stays stable/cacheable.
 *
 * pair, level: quadtree pair index + depth
 * tile: "x_y" grid location
 * side: 'he' (fixed) | 'ihc' (moving)
 * dx, dy: WSI-pixel offset baked into the crop
 * returns: request URL string
 */
export function liveCropUrl(
	pair: number,
	level: number,
	tile: string,
	side: 'he' | 'ihc',
	dx = 0,
	dy = 0
): string {
	const [x, y] = tile.split('_');
	let u = `/api/live-crop/tile?pair=${pair}&level=${level}&x=${x}&y=${y}&side=${side}`;
	if (dx !== 0 || dy !== 0) u += `&dx=${dx}&dy=${dy}`;
	return u;
}

/**
 * URL for the whole-image greyscale preview (raw, no deskew warp) at grid*CNN
 * resolution. `level` selects sharpness (0 = 512x344, 2 = 2048x1376 ≈ 4x). Used
 * by the deskew landmarking page.
 */
export function liveWholeUrl(pair: number, side: 'he' | 'ihc', level: number): string {
	return `/api/live-crop/whole?pair=${pair}&side=${side}&level=${level}`;
}

/**
 * URL for the per-tile FFT phase-correlation map PNG.
 *
 * The surface is computed on HE (fixed) vs IHC recropped at the prior base
 * (dx, dy), so the argmax reproduces the FFT residual. mx/my (optional) mark the
 * chosen peak — the residual that produced the current refinement vector
 * (ux - prior_dx, uy - prior_dy).
 *
 * pair, depth: quadtree pair index + depth
 * tile: "x_y" grid location
 * dx, dy: prior base offset baked into the moving IHC crop
 * mx, my: chosen-peak residual to highlight
 * returns: request URL string
 */
export function fftMapUrl(
	pair: number,
	depth: number,
	tile: string,
	dx = 0,
	dy = 0,
	mx?: number,
	my?: number
): string {
	let u = `/api/c2f/fft-map?pair=${pair}&depth=${depth}&tile=${tile}`;
	if (dx !== 0 || dy !== 0) u += `&dx=${dx}&dy=${dy}`;
	if (mx !== undefined && my !== undefined) u += `&mx=${mx}&my=${my}`;
	return u;
}
