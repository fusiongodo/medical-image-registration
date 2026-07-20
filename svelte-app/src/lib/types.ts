/** depth → valid */
export type PairValidation = Record<string, boolean>;

/** pair_id → PairValidation */
export type ValidationStore = Record<string, PairValidation>;

export interface TileMeta {
	tile: string;
}

export interface PairStatus {
	pairId: number;
	/** highest depth that passed; null if nothing evaluated yet */
	finalLevel: number | null;
	/** true = all evaluated levels passed, false = a failure was recorded, null = untouched */
	outcome: 'pass' | 'fail' | null;
}

export const MAX_DEPTH = 5;

/** CNN tile dimensions (mirrors conf.CNN_INPUT_WIDTH/HEIGHT). A whole-image
 * preview at quadtree level L is (2**L * CNN_WIDTH) x (2**L * CNN_HEIGHT). */
export const CNN_WIDTH = 512;
export const CNN_HEIGHT = 344;

export function deriveStatus(validation: ValidationStore, pairId: number): PairStatus {
	const pv = validation[String(pairId)];
	if (!pv || Object.keys(pv).length === 0) {
		return { pairId, finalLevel: null, outcome: null };
	}

	const depths = Object.keys(pv)
		.map(Number)
		.sort((a, b) => a - b);

	let finalLevel: number | null = null;
	for (const d of depths) {
		if (pv[String(d)]) {
			finalLevel = d;
		} else {
			return { pairId, finalLevel, outcome: 'fail' };
		}
	}

	return { pairId, finalLevel, outcome: 'pass' };
}

export function nextDepthForPair(validation: ValidationStore, pairId: number): number | null {
	const pv = validation[String(pairId)] ?? {};
	const failing = Object.entries(pv).find(([, v]) => !v);
	if (failing) return null;
	const evaluated = Object.keys(pv).map(Number);
	if (evaluated.length === 0) return 0;
	const next = Math.max(...evaluated) + 1;
	return next > MAX_DEPTH ? null : next;
}
