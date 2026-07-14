import { json, error } from '@sveltejs/kit';
import { readdirSync, readFileSync, existsSync } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';

const CROPPED = resolve('..', 'data', 'cropped');

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const depthDir = join(CROPPED, pair, `d${depth}`);
	if (!existsSync(depthDir)) return json({});

	const result: Record<string, { dx: number; dy: number }> = {};
	for (const tileId of readdirSync(depthDir)) {
		const file = join(depthDir, tileId, 'elastix', 'displacement.json');
		if (!existsSync(file)) continue;
		try {
			const { dx, dy } = JSON.parse(readFileSync(file, 'utf-8'));
			if (typeof dx === 'number' && typeof dy === 'number') {
				result[tileId] = { dx, dy };
			}
		} catch {
			// malformed file — skip silently
		}
	}
	return json(result);
};
