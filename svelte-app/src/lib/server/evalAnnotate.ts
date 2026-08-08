import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const LAYERS = ['he', 'ihc'] as const;

export type AnnotatePageData = {
	pairId: number;
	numPairs: number;
	fullReady: boolean;
	fullMeta: { w: number; h: number; qw: number; qh: number; nq?: number; scale?: number } | null;
	ready: boolean;
	mainSetId: string | null;
	mainSetName: string | null;
};

export function loadAnnotatePair(pairParam: string): AnnotatePageData | 'out_of_range' {
	const numPairs = pairCount();
	const pairId = Number(pairParam);
	if (!Number.isInteger(pairId) || pairId < 0 || pairId >= numPairs) {
		return 'out_of_range';
	}
	const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(pairId));
	const fullDir = resolve(dir, 'full');
	const metaPath = resolve(fullDir, 'meta.json');
	let fullMeta: AnnotatePageData['fullMeta'] = null;
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
	const activePath = resolve(
		REPO_ROOT,
		'data',
		'curated_field_sets',
		'fft',
		'tps',
		String(pairId),
		'active.json'
	);
	if (existsSync(activePath)) {
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
		}
	}

	const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
	return { pairId, numPairs, fullReady, fullMeta, ready, mainSetId, mainSetName };
}
