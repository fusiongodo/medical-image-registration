import { json, error } from '@sveltejs/kit';
import { readdirSync, existsSync, readFileSync } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';

const CROPPED      = resolve('..', 'data', 'cropped');
const N_BINS       = 100;
const CUTOFF_PCT   = 0.95;
const PATCH_SIZES  = [5, 10, 20, 30, 40, 50];

function _percentile(sorted: number[], p: number): number {
	if (sorted.length === 0) return 0;
	const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
	return sorted[idx];
}

export const GET: RequestHandler = ({ url }) => {
	const pair      = url.searchParams.get('pair');
	const depth     = url.searchParams.get('depth');
	const patchSize = url.searchParams.get('patchSize') ?? '50';
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const depthDir = join(CROPPED, pair, `d${depth}`);
	if (!existsSync(depthDir)) {
		return json({ bins: [], counts: [], maxVal: 1, totalTiles: 0, withDisplacement: 0, withMetrics: 0, missing: 0 });
	}

	const values: number[] = [];
	let totalTiles = 0;
	let withDisplacement = 0;

	for (const tileId of readdirSync(depthDir)) {
		const tileDir = join(depthDir, tileId);
		if (!existsSync(join(tileDir, 'he.png'))) continue;
		totalTiles++;

		if (!existsSync(join(tileDir, 'elastix', 'displacement.json'))) continue;
		withDisplacement++;

		const metricsPath = join(tileDir, 'metrics.json');
		if (!existsSync(metricsPath)) continue;

		try {
			const m = JSON.parse(readFileSync(metricsPath, 'utf-8'));
			const bp = (m.by_patch as Record<string, { lncc2_auto: number }> | undefined)?.[patchSize];
			if (typeof bp?.lncc2_auto !== 'number') continue;
			values.push(bp.lncc2_auto);
		} catch {
			// malformed file — skip
		}
	}

	const withMetrics = values.length;
	const missing     = withDisplacement - withMetrics;

	if (withMetrics === 0) {
		return json({ bins: [], counts: [], maxVal: 1, totalTiles, withDisplacement, withMetrics, missing });
	}

	const sorted   = [...values].sort((a, b) => a - b);
	const maxVal   = _percentile(sorted, CUTOFF_PCT);
	const binWidth = maxVal > 0 ? maxVal / N_BINS : 1 / N_BINS;

	const counts = new Array<number>(N_BINS).fill(0);
	for (const v of values) {
		const idx = Math.min(N_BINS - 1, Math.floor(v / binWidth));
		counts[Math.max(0, idx)]++;
	}

	const bins = Array.from({ length: N_BINS }, (_, i) =>
		parseFloat((i * binWidth).toFixed(6))
	);

	return json({ bins, counts, maxVal, totalTiles, withDisplacement, withMetrics, missing });
};
