/**
 * Typed fetch wrappers for the coarse-to-fine + registration-annotation API.
 *
 * Centralises endpoint paths, HTTP verbs and JSON plumbing so the page and the
 * C2F panel share one client. Callers keep their own stale-response guards
 * (pair/depth may change mid-flight) since only they know the current context.
 */

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export interface PatchEntry {
	lncc2: number;
	lncc2_auto: number;
	factor_auto: number;
}

export interface Candidate {
	tile_loc: string;
	u: number;
	v: number;
	psr: number;
	delta_px?: number;
	by_patch?: Record<string, PatchEntry>;
}

export interface CandidatesResponse {
	cached: boolean;
	candidates?: Candidate[];
}

export interface JobState {
	running: boolean;
	done: number;
	total: number;
	error: string | null;
	finishedAt: number | null;
}

export interface TileResult {
	tile_loc: string;
	psr: number;
	residual: number;
	kept: boolean;
	excluded?: boolean;
	masked?: boolean;
	annotated?: 'approve' | 'correct' | 'exclude' | null;
	dx: number;
	dy: number;
	ux: number;
	uy: number;
	prior_dx: number;
	prior_dy: number;
}

export interface RefitData {
	tau: number;
	kept: number;
	rejected: number;
	excluded?: number;
	masked?: number;
	n_human?: number;
	mean_residual: number;
	tiles: TileResult[];
}

export type Rating = 'bad' | 'ok' | 'good';

export interface FieldSet {
	id: string;
	name: string;
	saved_depth: number | null;
	tau?: number;
	n_human?: number;
	updated: number;
	rating?: Rating | null;
}

export interface FieldSetList {
	sets: FieldSet[];
	active: string | null;
	main: string | null;
}

export interface RegAnnotation {
	type: 'approve' | 'correct' | 'exclude';
	u: number;
	v: number;
}

/** Gate the fit either by absolute τ or by a keep-fraction (1 − exclude%). */
export type Gate = { tau: number } | { keep: number };

function gateQuery(gate: Gate): string {
	return 'keep' in gate ? `keep=${gate.keep}` : `tau=${gate.tau}`;
}

export async function getCandidates(pair: number, depth: number): Promise<CandidatesResponse> {
	const r = await fetch(`/api/c2f/candidates?pair=${pair}&depth=${depth}`);
	return r.json();
}

export async function computeCandidates(pair: number, depth: number): Promise<{ state: JobState }> {
	const r = await fetch('/api/c2f/candidates', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify({ pair_id: pair, depth })
	});
	return r.json();
}

export async function getProgress(pair: number, depth: number): Promise<JobState> {
	const r = await fetch(`/api/c2f/candidates/progress?pair=${pair}&depth=${depth}`);
	return r.json();
}

/** Add LNCC by_patch metrics to an already-cached candidate set (slow, opt-in). */
export async function computeMetrics(pair: number, depth: number): Promise<{ state: JobState }> {
	const r = await fetch('/api/c2f/metrics', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify({ pair_id: pair, depth })
	});
	return r.json();
}

export async function getMetricsProgress(pair: number, depth: number): Promise<JobState> {
	const r = await fetch(`/api/c2f/metrics/progress?pair=${pair}&depth=${depth}`);
	return r.json();
}

/** Refit result plus the error text branch the panel surfaces on failure. */
export async function getRefit(
	pair: number,
	depth: number,
	gate: Gate
): Promise<{ ok: boolean; data?: RefitData; error?: string }> {
	const r = await fetch(`/api/c2f/refit?pair=${pair}&depth=${depth}&${gateQuery(gate)}`);
	if (r.ok) return { ok: true, data: await r.json() };
	return { ok: false, error: (await r.text().catch(() => '')) || `refit failed (${r.status})` };
}

export async function saveFieldRequest(pair: number, depth: number, gate: Gate): Promise<boolean> {
	const body =
		'keep' in gate ? { pair_id: pair, depth, keep: gate.keep } : { pair_id: pair, depth, tau: gate.tau };
	const r = await fetch('/api/c2f/save-field', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify(body)
	});
	return r.ok;
}

/** Result of the refinement-aware FFT recompute over the current level's red tiles. */
export interface ResolveResult {
	ok: boolean;
	tau: number;
	tried: number;
	resolved: number;
	approved: number;
}

/**
 * Recompute FFT for all red (above-tau / excluded) tiles at pair+depth, choosing
 * the highest-PSR peak that aligns with the current refinement field. Overwrites
 * the resolved tiles' cached candidates; callers should refit afterwards.
 */
export async function resolveExcluded(
	pair: number,
	depth: number,
	gate: Gate
): Promise<ResolveResult | null> {
	const body =
		'keep' in gate ? { pair_id: pair, depth, keep: gate.keep } : { pair_id: pair, depth, tau: gate.tau };
	const r = await fetch('/api/c2f/resolve', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify(body)
	});
	if (!r.ok) return null;
	return r.json();
}

export async function getFieldSets(pair: number): Promise<FieldSetList> {
	const r = await fetch(`/api/c2f/field-set?pair=${pair}`);
	if (!r.ok) throw new Error(`field-set list failed (${r.status})`);
	return r.json();
}

export async function postFieldSet(
	pair: number,
	body: Record<string, unknown>
): Promise<{ ok?: boolean } | null> {
	try {
		const r = await fetch('/api/c2f/field-set', {
			method: 'POST',
			headers: JSON_HEADERS,
			body: JSON.stringify({ pair_id: pair, ...body })
		});
		if (!r.ok) return null;
		return await r.json();
	} catch {
		return null;
	}
}

export async function getRegAnnotations(pair: number, level: number): Promise<Record<string, RegAnnotation>> {
	const r = await fetch(`/api/c2f/annotate?pair=${pair}&level=${level}`);
	return r.json();
}

/** POST an approve/correct/exclude/clear vote. Returns the raw Response so the
 * caller can read res.ok / res.text() for its own flash messaging. */
export function postRegAnnotation(
	pair: number,
	level: number,
	tile: string,
	action: 'approve' | 'correct' | 'exclude' | 'clear',
	u = 0,
	v = 0
): Promise<Response> {
	const payload: Record<string, unknown> = { pair_id: pair, level, tile_loc: tile, action };
	if (action !== 'clear') {
		payload.u = u;
		payload.v = v;
	}
	return fetch('/api/c2f/annotate', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify(payload)
	});
}

/** A whole-image correspondence pair for the global deskew (level-0 crop pixels). */
export interface DeskewPoint {
	he: [number, number];
	ihc: [number, number];
}

export interface DeskewState {
	points: DeskewPoint[];
	depth: number | null;
}

export async function getDeskew(pair: number): Promise<DeskewState> {
	const r = await fetch(`/api/c2f/deskew?pair=${pair}`);
	if (!r.ok) return { points: [], depth: null };
	return r.json();
}

/** Fit + persist the global deskew from >=3 correspondence pairs; also writes the
 * smooth field and discards the pair's FFT candidate caches. */
export async function applyDeskew(pair: number, depth: number, points: DeskewPoint[]): Promise<boolean> {
	const r = await fetch('/api/c2f/deskew', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify({ pair_id: pair, depth, points, action: 'apply' })
	});
	return r.ok;
}

export async function clearDeskew(pair: number): Promise<boolean> {
	const r = await fetch('/api/c2f/deskew', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify({ pair_id: pair, action: 'clear' })
	});
	return r.ok;
}

/** Effective masked map for one level: { "x_y": true, ... } (masked only). */
export async function getMasks(pair: number, level: number): Promise<Record<string, true>> {
	const r = await fetch(`/api/c2f/mask?pair=${pair}&level=${level}`);
	return r.json();
}

/** Toggle a tile's mask-out state. Masks propagate forward to descendants;
 * `clear` drops an override, reverting to the inherited state. */
export function postMask(
	pair: number,
	level: number,
	tile: string,
	action: 'mask' | 'unmask' | 'clear'
): Promise<Response> {
	return fetch('/api/c2f/mask', {
		method: 'POST',
		headers: JSON_HEADERS,
		body: JSON.stringify({ pair_id: pair, level, tile_loc: tile, action })
	});
}
