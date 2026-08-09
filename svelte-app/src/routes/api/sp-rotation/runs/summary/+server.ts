import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { buildSummary } from '$lib/server/spRotStore';

export const GET: RequestHandler = async ({ url }) => {
	const run = url.searchParams.get('run');
	if (!run) error(400, 'Missing run');
	try {
		return json(buildSummary(run));
	} catch (e) {
		error(500, `summary failed: ${(e as Error).message}`);
	}
};
