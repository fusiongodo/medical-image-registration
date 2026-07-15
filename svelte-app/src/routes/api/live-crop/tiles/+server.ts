import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { cropRequest } from '$lib/liveCropWorker';

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const level = url.searchParams.get('level');
	if (!pair || !level || !/^\d+$/.test(pair) || !/^\d+$/.test(level)) {
		error(400, 'Missing or invalid pair / level');
	}

	try {
		const payload = await cropRequest({
			op: 'tiles',
			pair: Number(pair),
			level: Number(level)
		});
		return json(payload);
	} catch (err) {
		error(500, err instanceof Error ? err.message : 'live-crop worker failed');
	}
};
