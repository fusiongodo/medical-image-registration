import { error, json } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const LAYERS = new Set([
	'he',
	'ihc',
	'ihc_warped',
	'ihc_fieldset_tps',
	'ihc_fieldset_wendland'
]);

function fullDir(pairId: number) {
	return resolve(REPO_ROOT, 'data', 'regwsi', String(pairId), 'full');
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const layer = url.searchParams.get('layer');
	const metaOnly = url.searchParams.get('meta') === '1';

	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);

	const dir = fullDir(pairId);
	const metaPath = resolve(dir, 'meta.json');

	if (metaOnly) {
		if (!existsSync(metaPath)) {
			error(404, `no full-res mosaic for pair ${pairId}; run: python regWSI/make_full.py ${pairId}`);
		}
		return json(JSON.parse(readFileSync(metaPath, 'utf-8')));
	}

	const qx = url.searchParams.get('qx');
	const qy = url.searchParams.get('qy');
	if (!layer || !LAYERS.has(layer)) {
		error(400, 'layer must be he, ihc, ihc_warped, ihc_fieldset_tps, or ihc_fieldset_wendland');
	}
	if (qx == null || qy == null || !/^\d+$/.test(qx) || !/^\d+$/.test(qy)) {
		error(400, 'qx and qy must be non-negative integers');
	}

	const file = resolve(dir, `${layer}_y${qy}_x${qx}.jpg`);
	if (!existsSync(file)) {
		error(404, `no full-res mosaic for pair ${pairId}; run: python regWSI/make_full.py ${pairId}`);
	}

	const buffer = readFileSync(file);
	return new Response(buffer, {
		headers: { 'Content-Type': 'image/jpeg', 'Cache-Control': 'no-store' }
	});
};
