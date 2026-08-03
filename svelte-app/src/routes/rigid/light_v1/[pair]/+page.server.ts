import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { pairCount } from '$lib/server/pairs';

export const load: PageServerLoad = ({ params }) => {
	const numPairs = pairCount();
	const pairId = Number(params.pair);
	if (!Number.isInteger(pairId) || pairId < 0 || pairId >= numPairs) {
		error(404, `pair ${params.pair} out of range`);
	}
	return { pairId, numPairs };
};
