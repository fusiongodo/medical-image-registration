import { json, error } from '@sveltejs/kit';
import { pairCount } from '$lib/server/pairs';
import { jobs, rigidJobKey } from '$lib/c2fJobs';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = ({ url }) => {
	const pair = Number(url.searchParams.get('pair'));
	if (!Number.isInteger(pair) || pair < 0 || pair >= pairCount()) {
		error(400, 'Missing/invalid pair');
	}
	const state = jobs.get(rigidJobKey(pair));
	return json(state ?? { running: false, done: 0, total: 0, error: null, finishedAt: null, stage: null });
};
