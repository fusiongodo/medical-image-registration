import type { PageServerLoad } from './$types';
import { pairCount } from '$lib/server/pairs';
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

export const load: PageServerLoad = () => {
	const n = pairCount();
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(i));
		const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		const { mainSetId, mainSetName } = mainSetMeta(i);
		pairs.push({
			pairId: i,
			ready,
			landmarkCount: landmarkCount(dir),
			mainSetId,
			mainSetName
		});
	}
	return { pairs };
};
