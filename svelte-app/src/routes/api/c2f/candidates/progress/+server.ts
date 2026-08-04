import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { jobs, jobKey } from '$lib/c2fJobs';

const LAMS = new Set(['fft', 'superpoint_glue']);

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	const lamRaw = url.searchParams.get('lam');
	const lam = lamRaw && LAMS.has(lamRaw) ? lamRaw : 'fft';
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const key = jobKey(parseInt(pair, 10), parseInt(depth, 10), 'candidates', lam);
	const state = jobs.get(key) ?? null;
	return json(state);
};
