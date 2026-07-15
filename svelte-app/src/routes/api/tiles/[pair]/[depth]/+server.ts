import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import type { TileMeta } from '$lib/types';
import { cropRequest } from '$lib/liveCropWorker';

export const GET: RequestHandler = async ({ params }) => {
	const { pair, depth } = params;
	if (!/^\d+$/.test(pair) || !/^\d+$/.test(depth)) {
		error(400, 'Invalid pair or depth');
	}

	try {
		const payload = await cropRequest({ op: 'tiles', pair: Number(pair), level: Number(depth) });
		const tiles: TileMeta[] = ((payload.tiles as string[]) ?? []).map((tile) => ({ tile }));
		return json(tiles);
	} catch {
		return json([]);
	}
};
