/**
 * Displacement resolution shared by the tile grid and the C2F panel.
 *
 * All vectors are WSI-pixel offsets applied on top of a per-tile `base`
 * (the previous level's saved field, or 0 at the base level). Human input
 * (landmark points / stored correction votes) overrides the FFT auto vector.
 */

export interface Vec {
	dx: number;
	dy: number;
}

export interface Point {
	x: number;
	y: number;
}

export interface TileAnnotation {
	hePoints: Point[];
	ihcPoints: Point[];
}

export interface RegAnnotation {
	type: 'approve' | 'correct' | 'exclude';
	u: number;
	v: number;
}

/**
 * Mean landmark displacement over all placed HE/IHC point pairs, on top of base.
 * ann: placed points (may be undefined); base: previous-level offset.
 * returns: {dx,dy} or null when no complete point pair exists.
 */
export function manualDisplacement(ann: TileAnnotation | undefined, base: Vec): Vec | null {
	if (!ann) return null;
	const pairs = Math.min(ann.hePoints.length, ann.ihcPoints.length);
	if (pairs === 0) return null;
	let dx = 0;
	let dy = 0;
	for (let i = 0; i < pairs; i++) {
		dx += ann.hePoints[i].x - ann.ihcPoints[i].x;
		dy += ann.hePoints[i].y - ann.ihcPoints[i].y;
	}
	return { dx: base.dx + dx / pairs, dy: base.dy + dy / pairs };
}

/**
 * Correction vector from the most recently placed HE/IHC point pair, on top of
 * base. Used when committing a `correct` vote.
 * returns: {dx,dy} or null when fewer than one point on either side.
 */
export function correctVector(ann: TileAnnotation | undefined, base: Vec): Vec | null {
	if (!ann || ann.hePoints.length < 1 || ann.ihcPoints.length < 1) return null;
	const he = ann.hePoints[ann.hePoints.length - 1];
	const ihc = ann.ihcPoints[ann.ihcPoints.length - 1];
	return { dx: base.dx + he.x - ihc.x, dy: base.dy + he.y - ihc.y };
}

/**
 * Human-driven displacement for a tile: live landmark drag wins, else a stored
 * `correct` vote's vector. Returns null when the tile has no human input (the
 * caller decides whether to fall back to the FFT auto vector).
 */
export function humanVec(manual: Vec | null, corr: RegAnnotation | undefined): Vec | null {
	if (manual) return manual;
	if (corr?.type === 'correct') return { dx: corr.u, dy: corr.v };
	return null;
}
