import { error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { PageServerLoad } from './$types';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const LAYERS = ['he', 'ihc'] as const;

export const load: PageServerLoad = ({ params }) => {
	const numPairs = pairCount();
	const pairId = Number(params.pair);
	if (!Number.isInteger(pairId) || pairId < 0 || pairId >= numPairs) {
		error(404, `pair ${params.pair} out of range`);
	}
	const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(pairId));
	const fullDir = resolve(dir, 'full');
	const metaPath = resolve(fullDir, 'meta.json');
	let fullMeta: { w: number; h: number; qw: number; qh: number; nq?: number; scale?: number } | null =
		null;
	if (existsSync(metaPath)) {
		try {
			fullMeta = JSON.parse(readFileSync(metaPath, 'utf-8'));
		} catch {
			fullMeta = null;
		}
	}
	const nq = fullMeta?.nq ?? 2;
	const fullReady =
		!!fullMeta &&
		LAYERS.every((layer) => {
			for (let qy = 0; qy < nq; qy++) {
				for (let qx = 0; qx < nq; qx++) {
					if (!existsSync(resolve(fullDir, `${layer}_y${qy}_x${qx}.jpg`))) return false;
				}
			}
			return true;
		});

	let mainSetId: string | null = null;
	let mainSetName: string | null = null;
	const activePath = resolve(REPO_ROOT, 'data', 'field_sets', String(pairId), 'active.json');
	if (existsSync(activePath)) {
		try {
			const active = JSON.parse(readFileSync(activePath, 'utf-8'));
			mainSetId = active.main_set_id ?? active.set_id ?? null;
			if (mainSetId) {
				const manifestPath = resolve(
					REPO_ROOT,
					'data',
					'field_sets',
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
		}
	}

	const previewReady =
		existsSync(resolve(dir, 'preview', 'he.png')) &&
		existsSync(resolve(dir, 'preview', 'ihc_warped.png')) &&
		existsSync(resolve(dir, 'out', 'displacement_field.mha'));

	return { pairId, numPairs, fullReady, fullMeta, ready: previewReady, mainSetId, mainSetName };
};
