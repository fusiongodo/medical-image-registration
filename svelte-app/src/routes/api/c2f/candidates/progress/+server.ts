import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { jobs, jobKey } from '$lib/c2fJobs';

export const GET: RequestHandler = ({ url }) => {
	const pair  = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const key   = jobKey(parseInt(pair, 10), parseInt(depth, 10));
	const state = jobs.get(key) ?? null;
	return json(state);
};
