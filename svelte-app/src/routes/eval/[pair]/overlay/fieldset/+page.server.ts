import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { loadOverlayPair } from '$lib/server/evalOverlay';

const EST = new Set(['tps', 'wendland', 'bspline']);
const LAMS = new Set(['fft', 'superpoint_glue']);

export const load: PageServerLoad = ({ params, url }) => {
	const data = loadOverlayPair(params.pair);
	if (data.error === 'out_of_range') error(404, `pair ${params.pair} out of range`);
	const rawEst = url.searchParams.get('estimator') ?? 'tps';
	const estimator = EST.has(rawEst) ? rawEst : 'tps';
	const rawLam = url.searchParams.get('lam') ?? 'fft';
	const lam = LAMS.has(rawLam) ? rawLam : 'fft';
	const batch = url.searchParams.get('batch');
	return {
		...data,
		estimator,
		lam,
		batch: batch && batch.length ? batch : null
	};
};
