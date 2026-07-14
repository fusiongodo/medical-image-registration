import { error } from '@sveltejs/kit';
import { readdirSync, existsSync, type Dirent } from 'fs';
import { resolve, join } from 'path';
import type { PageServerLoad } from './$types';
import type { TileMeta } from '$lib/types';
import { MAX_DEPTH, NUM_PAIRS } from '$lib/types';

const CROPPED_ROOT        = resolve('..', 'data', 'cropped');
const CROPPED_SMOOTH_ROOT = resolve('..', 'data', 'cropped_smooth');

export const load: PageServerLoad = ({ params, url }) => {
	const pairId = parseInt(params.pair, 10);
	const depth  = parseInt(params.depth, 10);
	const smooth = url.searchParams.get('source') === 'smooth';

	if (isNaN(pairId) || pairId < 0 || pairId >= NUM_PAIRS) error(404, 'Invalid pair');
	if (isNaN(depth)  || depth  < 0 || depth  > MAX_DEPTH)  error(404, 'Invalid depth');

	const root     = smooth ? CROPPED_SMOOTH_ROOT : CROPPED_ROOT;
	const dirLabel = smooth ? 'cropped_smooth' : 'cropped';
	const depthDir = join(root, String(pairId), `d${depth}`);

	const smoothDepthDir   = join(CROPPED_SMOOTH_ROOT, String(pairId), `d${depth}`);
	const smoothAvailable  = existsSync(smoothDepthDir);

	let tiles: TileMeta[] = [];

	if (existsSync(depthDir)) {
		tiles = readdirSync(depthDir, { withFileTypes: true })
			.filter((e: Dirent) => e.isDirectory())
			.map((e: Dirent) => ({
				tile: e.name,
				he:  `data/${dirLabel}/${pairId}/d${depth}/${e.name}/he.png`,
				ihc: `data/${dirLabel}/${pairId}/d${depth}/${e.name}/ihc.png`,
			}))
			.sort((a: TileMeta, b: TileMeta) => {
				const [ax, ay] = a.tile.split('_').map(Number);
				const [bx, by] = b.tile.split('_').map(Number);
				return ay !== by ? ay - by : ax - bx;
			});
	}

	return { pairId, depth, tiles, smooth, smoothAvailable };
};
