import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');

export type FullMeta = { w: number; h: number; qw: number; qh: number; nq?: number; scale?: number };

export function layersReady(fullDir: string, nq: number, layers: readonly string[]): boolean {
	return layers.every((layer) => {
		for (let qy = 0; qy < nq; qy++) {
			for (let qx = 0; qx < nq; qx++) {
				if (!existsSync(resolve(fullDir, `${layer}_y${qy}_x${qx}.jpg`))) return false;
			}
		}
		return true;
	});
}

export function loadOverlayPair(pairParam: string) {
	const numPairs = pairCount();
	const pairId = Number(pairParam);
	if (!Number.isInteger(pairId) || pairId < 0 || pairId >= numPairs) {
		return { error: 'out_of_range' as const, pairId, numPairs };
	}
	const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(pairId));
	const fullDir = resolve(dir, 'full');
	const metaPath = resolve(fullDir, 'meta.json');
	let fullMeta: FullMeta | null = null;
	if (existsSync(metaPath)) {
		try {
			fullMeta = JSON.parse(readFileSync(metaPath, 'utf-8'));
		} catch {
			fullMeta = null;
		}
	}
	const nq = fullMeta?.nq ?? 2;
	const heReady = !!fullMeta && layersReady(fullDir, nq, ['he']);
	const warpedReady = heReady && layersReady(fullDir, nq, ['ihc_warped']);
	const fieldsetTpsReady = heReady && layersReady(fullDir, nq, ['ihc_fieldset_tps']);
	const fieldsetWendlandReady = heReady && layersReady(fullDir, nq, ['ihc_fieldset_wendland']);
	const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));

	return {
		error: null as null,
		pairId,
		numPairs,
		fullMeta,
		heReady,
		warpedReady,
		fieldsetTpsReady,
		fieldsetWendlandReady,
		ready,
		nq
	};
}
