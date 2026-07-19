import type { LayoutServerLoad } from './$types';
import type { ValidationStore } from '$lib/types';
import { MAX_DEPTH, NUM_PAIRS } from '$lib/types';
import { readFileSync, existsSync } from 'fs';
import { resolve, join } from 'path';

const DATA_ROOT       = resolve('..', 'data');
const VALIDATION_PATH = join(DATA_ROOT, 'quadtree_level_validation.json');
const SMOOTH_DIR      = join(DATA_ROOT, 'smooth_c2f');
const CACHE_DIR       = join(DATA_ROOT, 'c2f_cache');
const FIELD_SETS_DIR  = join(DATA_ROOT, 'field_sets');

type Rating = 'bad' | 'ok' | 'good';

/** Rating of each pair's pinned main field set (only for pairs that have one). */
function pairRatings(): Record<number, Rating> {
	const out: Record<number, Rating> = {};
	for (let pair = 0; pair < NUM_PAIRS; pair++) {
		const activePath = join(FIELD_SETS_DIR, String(pair), 'active.json');
		if (!existsSync(activePath)) continue;
		try {
			const mainId = JSON.parse(readFileSync(activePath, 'utf-8')).main_set_id;
			if (!mainId) continue;
			const manifestPath = join(FIELD_SETS_DIR, String(pair), mainId, 'manifest.json');
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
function fieldCompletePairs(): number[] {
	const out: number[] = [];
	for (let pair = 0; pair < NUM_PAIRS; pair++) {
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
	return { validation, fieldComplete: fieldCompletePairs(), ratings: pairRatings() };
};
