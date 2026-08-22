import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { loadOverlayPair } from '$lib/server/evalOverlay';

export const load: PageServerLoad = ({ params, url }) => {
	const side = params.side;
	if (side !== 'he' && side !== 'ihc') error(404, 'side must be he or ihc');
	const data = loadOverlayPair(params.pair, url.searchParams.get('dataset'));
	if (data.error === 'out_of_range') error(404, `pair ${params.pair} out of range`);
	const batch = url.searchParams.get('batch');
	return {
		...data,
		side,
		batch: batch && batch.length ? batch : null
	};
};
