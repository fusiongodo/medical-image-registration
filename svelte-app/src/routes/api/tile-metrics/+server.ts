import { json, error } from '@sveltejs/kit';
import { readdirSync, readFileSync, existsSync } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';

const CROPPED = resolve('..', 'data', 'cropped');

interface PatchEntry {
	lncc2: number;
	lncc2_auto: number;
	factor_auto: number;
}

interface TileMetrics {
	delta_px: number;
	dx: number;
	dy: number;
	by_patch: Record<string, PatchEntry>;
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
			if (typeof m.delta_px === 'number' && m.by_patch && typeof m.by_patch === 'object') {
				result[tileId] = {
					delta_px: m.delta_px,
					dx: typeof m.dx === 'number' ? m.dx : 0,
					dy: typeof m.dy === 'number' ? m.dy : 0,
					by_patch: m.by_patch,
				};
			}
		} catch {
			// malformed file — skip
		}
	}
	return json(result);
};
