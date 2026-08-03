import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
import { loadOverlayPair } from '$lib/server/regwsiOverlay';

export const load: PageServerLoad = ({ params, url }) => {
	const data = loadOverlayPair(params.pair);
	if (data.error === 'out_of_range') error(404, `pair ${params.pair} out of range`);
	const raw = url.searchParams.get('estimator');
	let estimator: 'tps' | 'wendland' = raw === 'wendland' ? 'wendland' : 'tps';
	if (raw !== 'tps' && raw !== 'wendland') {
		if (!data.fieldsetTpsReady && data.fieldsetWendlandReady) estimator = 'wendland';
	}
	return { ...data, estimator };
};
