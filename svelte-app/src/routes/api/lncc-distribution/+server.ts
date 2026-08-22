import { json, error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';

const CACHE_DIR  = resolve('..', 'data', 'c2f_cache');
const N_BINS     = 100;
const CUTOFF_PCT = 0.95;

function _percentile(sorted: number[], p: number): number {
	if (sorted.length === 0) return 0;
	const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
	return sorted[idx];
}

export const GET: RequestHandler = ({ url }) => {
	const pair      = url.searchParams.get('pair');
	const depth     = url.searchParams.get('depth');
	const patchSize = url.searchParams.get('patchSize') ?? '50';
	const lam       = url.searchParams.get('lam') ?? 'fft';
	const estimator = url.searchParams.get('estimator') ?? 'tps';
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const empty = { bins: [], counts: [], maxVal: 1, totalTiles: 0, withDisplacement: 0, withMetrics: 0, missing: 0 };

	const cachePath = join(CACHE_DIR, lam, estimator, `${pair}_d${depth}.json`);
	if (!existsSync(cachePath)) return json(empty);

	let candidates: { by_patch?: Record<string, { lncc2_auto: number }> }[] = [];
	try {
		candidates = JSON.parse(readFileSync(cachePath, 'utf-8')).candidates ?? [];
	} catch {
		return json(empty);
	}

	const values: number[] = [];
	for (const c of candidates) {
		const v = c.by_patch?.[patchSize]?.lncc2_auto;
		if (typeof v === 'number') values.push(v);
	}

	const withMetrics = values.length;
	const totalTiles = candidates.length;
	if (withMetrics === 0) {
		return json({ ...empty, totalTiles, withDisplacement: totalTiles });
	}

	const sorted   = [...values].sort((a, b) => a - b);
	const maxVal   = _percentile(sorted, CUTOFF_PCT);
	const binWidth = maxVal > 0 ? maxVal / N_BINS : 1 / N_BINS;

	const counts = new Array<number>(N_BINS).fill(0);
	for (const v of values) {
		const idx = Math.min(N_BINS - 1, Math.floor(v / binWidth));
		counts[Math.max(0, idx)]++;
	}
	const bins = Array.from({ length: N_BINS }, (_, i) => parseFloat((i * binWidth).toFixed(6)));

	return json({
		bins, counts, maxVal,
		totalTiles,
		withDisplacement: totalTiles,
		withMetrics,
		missing: 0,
	});
};
