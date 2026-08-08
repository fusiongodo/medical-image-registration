import type { PageServerLoad } from './$types';
import { listDatasets, normalizeDataset, pairCount, regwsiRoot } from '$lib/server/datasets';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');

function landmarkCount(dir: string): number {
	const path = resolve(dir, 'landmarks.json');
	if (!existsSync(path)) return 0;
	try {
		const data = JSON.parse(readFileSync(path, 'utf-8'));
		return Array.isArray(data.points) ? data.points.length : 0;
	} catch {
		return 0;
	}
}

function mainSetMeta(pairId: number): { mainSetId: string | null; mainSetName: string | null } {
	let mainSetId: string | null = null;
	let mainSetName: string | null = null;
	const activePath = resolve(
		REPO_ROOT,
		'data',
		'curated_field_sets',
		'fft',
		'tps',
		String(pairId),
		'active.json'
	);
	if (!existsSync(activePath)) return { mainSetId, mainSetName };
	try {
		const active = JSON.parse(readFileSync(activePath, 'utf-8'));
		mainSetId = active.main_set_id ?? active.set_id ?? null;
		if (mainSetId) {
			const manifestPath = resolve(
				REPO_ROOT,
				'data',
				'curated_field_sets',
				'fft',
				'tps',
				String(pairId),
				mainSetId,
				'manifest.json'
			);
			if (existsSync(manifestPath)) {
				const m = JSON.parse(readFileSync(manifestPath, 'utf-8'));
				mainSetName = m.name ?? mainSetId;
			} else {
				mainSetName = mainSetId;
			}
		}
	} catch {
		mainSetId = null;
		mainSetName = null;
	}
	return { mainSetId, mainSetName };
}

export const load: PageServerLoad = ({ url }) => {
	const dataset = normalizeDataset(url.searchParams.get('dataset'));
	const n = pairCount(dataset);
	const root = regwsiRoot(dataset);
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(root, String(i));
		const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		const { mainSetId, mainSetName } = dataset === 'muromi' ? mainSetMeta(i) : { mainSetId: null, mainSetName: null };
		pairs.push({
			pairId: i,
			ready,
			landmarkCount: landmarkCount(dir),
			mainSetId,
			mainSetName
		});
	}
	return { pairs, dataset, datasets: listDatasets() };
};
