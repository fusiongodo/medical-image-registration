import { json } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { datasetFromUrl, pairCount, pairDir } from '$lib/server/datasets';
import { fingerprintMatches, type PairIdentity } from '$lib/server/pairs';

function pairStatus(dataset: ReturnType<typeof datasetFromUrl>, pairId: number) {
	const dir = pairDir(dataset, pairId);
	const previewHe = resolve(dir, 'preview', 'he.png');
	const previewIhc = resolve(dir, 'preview', 'ihc_warped.png');
	const field = resolve(dir, 'out', 'displacement_field.mha');
	const metaPath = resolve(dir, 'meta.json');
	let meta: { identity?: PairIdentity; timestamp?: string; params?: string } | null = null;
	if (existsSync(metaPath)) {
		try {
			meta = JSON.parse(readFileSync(metaPath, 'utf-8'));
		} catch {
			meta = null;
		}
	}
	const ready = existsSync(previewHe) && existsSync(previewIhc) && existsSync(field);
	const identityOk =
		dataset === 'acrobat' ? true : fingerprintMatches(pairId, meta?.identity ?? null);
	return {
		pairId,
		dataset,
		ready,
		identityOk,
		timestamp: meta?.timestamp ?? null,
		params: meta?.params ?? null
	};
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const dataset = datasetFromUrl(url);
	const n = pairCount(dataset);
	if (pair != null) {
		if (!/^\d+$/.test(pair)) return json({ error: 'invalid pair' }, { status: 400 });
		const pairId = Number(pair);
		if (pairId < 0 || pairId >= n) return json({ error: 'out of range' }, { status: 404 });
		return json(pairStatus(dataset, pairId));
	}

	const pairs = [];
	for (let i = 0; i < n; i++) pairs.push(pairStatus(dataset, i));
	return json({ dataset, pairs });
};
