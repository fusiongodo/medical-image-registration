import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { matrixStatus } from '$lib/server/spRotStore';

export const GET: RequestHandler = async ({ url }) => {
	const run = url.searchParams.get('run');
	if (!run) error(400, 'Missing run');
	try {
		return json(matrixStatus(run));
	} catch (e) {
		error(500, `status failed: ${(e as Error).message}`);
	}
};
