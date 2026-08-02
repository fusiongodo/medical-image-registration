import type { LayoutServerLoad } from './$types';
import type { ValidationStore } from '$lib/types';
import { MAX_DEPTH } from '$lib/types';
import { pairCount } from '$lib/server/pairs';
import { readFileSync, existsSync } from 'fs';
import { resolve, join } from 'path';

const DATA_ROOT       = resolve('..', 'data');
const VALIDATION_PATH = join(DATA_ROOT, 'quadtree_level_validation.json');
const SMOOTH_DIR      = join(DATA_ROOT, 'smooth_c2f');
const CACHE_DIR       = join(DATA_ROOT, 'c2f_cache');
const FIELD_SETS_DIR  = join(DATA_ROOT, 'curated_field_sets', 'fft', 'tps');

type Rating = 'bad' | 'ok' | 'good';

/**
 * Rating surfaced in the sidebar for each pair. Prefers the pinned "main" set,
 * but falls back to the active set so that rating a set is visible immediately
 * without also having to pin it as main.
 */
function pairRatings(n: number): Record<number, Rating> {
	const out: Record<number, Rating> = {};
	for (let pair = 0; pair < n; pair++) {
		const activePath = join(FIELD_SETS_DIR, String(pair), 'active.json');
		if (!existsSync(activePath)) continue;
		try {
			const active = JSON.parse(readFileSync(activePath, 'utf-8'));
			const setId = active.main_set_id ?? active.set_id;
			if (!setId) continue;
			const manifestPath = join(FIELD_SETS_DIR, String(pair), setId, 'manifest.json');
			if (!existsSync(manifestPath)) continue;
			const rating = JSON.parse(readFileSync(manifestPath, 'utf-8')).rating;
			if (rating === 'bad' || rating === 'ok' || rating === 'good') out[pair] = rating;
		} catch {
			/* skip malformed */
		}
	}
	return out;
}

/** Pairs whose smooth field was saved at the deepest level (registration complete). */
function fieldCompletePairs(n: number): number[] {
	const out: number[] = [];
	for (let pair = 0; pair < n; pair++) {
		const fieldPath = join(SMOOTH_DIR, `${pair}_smooth_field.json`);
		if (!existsSync(fieldPath)) continue;

		let savedDepth: number | null = null;
		try {
			savedDepth = JSON.parse(readFileSync(fieldPath, 'utf-8')).saved_depth ?? null;
		} catch {
			savedDepth = null;
		}

		// Legacy fields predate `saved_depth`; fall back to the presence of the
		// deepest-level candidate cache, which is only produced once L5 is reached.
		const complete =
			savedDepth != null
				? savedDepth >= MAX_DEPTH
				: existsSync(join(CACHE_DIR, `${pair}_d${MAX_DEPTH}.json`));

		if (complete) out.push(pair);
	}
	return out;
}

export const load: LayoutServerLoad = () => {
	let validation: ValidationStore = {};
	if (existsSync(VALIDATION_PATH)) {
		try {
			validation = JSON.parse(readFileSync(VALIDATION_PATH, 'utf-8'));
		} catch {
			validation = {};
		}
	}
	const numPairs = pairCount();
	return {
		validation,
		numPairs,
		fieldComplete: fieldCompletePairs(numPairs),
		ratings: pairRatings(numPairs)
	};
};
