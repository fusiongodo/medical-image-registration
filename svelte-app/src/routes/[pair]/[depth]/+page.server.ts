import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import type { TileMeta } from '$lib/types';
import { MAX_DEPTH } from '$lib/types';
import { pairCount } from '$lib/server/pairs';
import { cropRequest } from '$lib/liveCropWorker';

export const load: PageServerLoad = async ({ params }) => {
	const pairId = parseInt(params.pair, 10);
	const depth  = parseInt(params.depth, 10);

	if (isNaN(pairId) || pairId < 0 || pairId >= pairCount()) error(404, 'Invalid pair');
	if (isNaN(depth)  || depth  < 0 || depth  > MAX_DEPTH)  error(404, 'Invalid depth');

	let tiles: TileMeta[] = [];
	try {
		const payload = await cropRequest({ op: 'tiles', pair: pairId, level: depth });
		const locs = (payload.tiles as string[]) ?? [];
		tiles = locs.map((tile) => ({ tile }));
	} catch {
		tiles = [];
	}

	return { pairId, depth, tiles };
};
