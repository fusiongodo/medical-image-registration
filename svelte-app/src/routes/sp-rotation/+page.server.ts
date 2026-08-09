import type { PageServerLoad } from './$types';
import { listDatasets, normalizeDataset, pairCount, regwsiRoot } from '$lib/server/datasets';
import { existsSync } from 'fs';
import { resolve } from 'path';

export const load: PageServerLoad = ({ url }) => {
	const dataset = normalizeDataset(url.searchParams.get('dataset') || 'muromi');
	const n = pairCount(dataset);
	const root = regwsiRoot(dataset);
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(root, String(i));
		const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		pairs.push({ pairId: i, ready });
	}
	return { pairs, dataset, datasets: listDatasets() };
};
