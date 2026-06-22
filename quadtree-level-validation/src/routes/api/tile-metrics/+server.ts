import { json, error } from '@sveltejs/kit';
import { readdirSync, readFileSync, existsSync } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';

const CROPPED = resolve('..', 'data', 'cropped');

interface TileMetrics {
	lncc2: number;
	lncc2_auto: number;
	delta_px: number;
	factor_auto: number;
}

export const GET: RequestHandler = ({ url }) => {
	const pair  = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const depthDir = join(CROPPED, pair, `d${depth}`);
	if (!existsSync(depthDir)) return json({});

	const result: Record<string, TileMetrics> = {};
	for (const tileId of readdirSync(depthDir)) {
		const file = join(depthDir, tileId, 'metrics.json');
		if (!existsSync(file)) continue;
		try {
			const m = JSON.parse(readFileSync(file, 'utf-8'));
			if (typeof m.lncc2 === 'number' && typeof m.lncc2_auto === 'number') {
				result[tileId] = {
					lncc2:       m.lncc2,
					lncc2_auto:  m.lncc2_auto,
					delta_px:    m.delta_px,
					factor_auto: m.factor_auto,
				};
			}
		} catch {
			// malformed file — skip
		}
	}
	return json(result);
};
