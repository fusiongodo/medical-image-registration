import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { loadOverlayPair } from '$lib/server/evalOverlay';

export const load: PageServerLoad = ({ params, url }) => {
	const data = loadOverlayPair(params.pair, url.searchParams.get('dataset'));
	if (data.error === 'out_of_range') error(404, `pair ${params.pair} out of range`);
	const batch = url.searchParams.get('batch');
	const rawEst = url.searchParams.get('estimator') ?? 'wendland';
	const estimator = ['tps', 'wendland', 'bspline'].includes(rawEst) ? rawEst : 'wendland';
	return {
		...data,
		estimator,
		batch: batch && batch.length ? batch : null
	};
};
