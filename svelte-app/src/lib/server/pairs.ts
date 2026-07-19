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
