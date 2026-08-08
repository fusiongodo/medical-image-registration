import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { loadAnnotatePair } from '$lib/server/evalAnnotate';

export const load: PageServerLoad = ({ params, url }) => {
	const data = loadAnnotatePair(params.pair, url.searchParams.get('dataset'));
	if (data === 'out_of_range') error(404, `pair ${params.pair} out of range`);
	return data;
};
