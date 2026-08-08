import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { batchJobs, batchJobKey } from '$lib/evalJobs';

export const GET: RequestHandler = ({ url }) => {
	const batch = url.searchParams.get('batch');
	if (!batch) error(400, 'Missing batch');
	const state = batchJobs.get(batchJobKey(batch)) ?? null;
	return json(state);
};
