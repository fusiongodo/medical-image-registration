import { json, error } from '@sveltejs/kit';
import { readdirSync, existsSync, type Dirent } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';
import type { TileMeta } from '$lib/types';

const CROPPED_ROOT = resolve('..', 'data', 'cropped');

export const GET: RequestHandler = ({ params }) => {
	const { pair, depth } = params;

	if (!/^\d+$/.test(pair) || !/^\d+$/.test(depth)) {
		error(400, 'Invalid pair or depth');
	}

	const depthDir = join(CROPPED_ROOT, pair, `d${depth}`);

	if (!existsSync(depthDir)) {
		return json([]);
	}

	const tiles: TileMeta[] = readdirSync(depthDir, { withFileTypes: true })
		.filter((e: Dirent) => e.isDirectory())
		.map((e: Dirent) => ({
			tile: e.name,
			he: `data/cropped/${pair}/d${depth}/${e.name}/he.png`,
			ihc: `data/cropped/${pair}/d${depth}/${e.name}/ihc.png`
		}))
		.sort((a: TileMeta, b: TileMeta) => {
			const [ax, ay] = a.tile.split('_').map(Number);
			const [bx, by] = b.tile.split('_').map(Number);
			return ay !== by ? ay - by : ax - bx;
		});

	return json(tiles);
};
