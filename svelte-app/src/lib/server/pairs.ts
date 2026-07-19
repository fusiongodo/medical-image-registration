import { readFileSync } from 'fs';
import { resolve } from 'path';

const LABELS_PATH = resolve('..', 'data', 'macos_labels.json');
const FALLBACK_PAIR_COUNT = 8;

/** Number of registration pairs, derived from the labels file (single source of truth). */
export function pairCount(): number {
	try {
		const json = JSON.parse(readFileSync(LABELS_PATH, 'utf-8'));
		return Array.isArray(json) && json.length > 0 ? json.length : FALLBACK_PAIR_COUNT;
	} catch {
		return FALLBACK_PAIR_COUNT;
	}
}

export interface PairIdentity {
	target_image_id: number;
	source_image_id: number;
}

/** Image-id fingerprint for a pair from the current labels, or null if unavailable. */
export function pairFingerprint(pair: number): PairIdentity | null {
	try {
		const json = JSON.parse(readFileSync(LABELS_PATH, 'utf-8'));
		const item = json?.[pair];
		if (!item) return null;
		return { target_image_id: item.target_image_id, source_image_id: item.source_image_id };
	} catch {
		return null;
	}
}

/**
 * True when a stored artifact fingerprint still matches the current labels.
 * A missing/legacy fingerprint is tolerated (returns true) so pre-existing
 * artifacts keep loading unchanged; a real mismatch signals the pair index has
 * drifted (e.g. labels were regenerated) and the artifact belongs to other images.
 */
export function fingerprintMatches(pair: number, stored: Partial<PairIdentity> | null | undefined): boolean {
	if (!stored || stored.target_image_id == null) return true;
	const fp = pairFingerprint(pair);
	if (!fp) return true;
	return stored.target_image_id === fp.target_image_id && stored.source_image_id === fp.source_image_id;
}
