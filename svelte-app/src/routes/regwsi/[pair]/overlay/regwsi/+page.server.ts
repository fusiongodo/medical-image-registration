import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { loadOverlayPair } from '$lib/server/regwsiOverlay';

export const load: PageServerLoad = ({ params }) => {
	const data = loadOverlayPair(params.pair);
	if (data.error === 'out_of_range') error(404, `pair ${params.pair} out of range`);
	return data;
};
