import { error } from '@sveltejs/kit';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { datasetFromUrl, pairCount, pairDir } from '$lib/server/datasets';

const LAYERS = new Set(['he', 'ihc_warped']);

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const layer = url.searchParams.get('layer');
	const dataset = datasetFromUrl(url);
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	if (!layer || !LAYERS.has(layer)) error(400, 'layer must be he or ihc_warped');

	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount(dataset)) error(404, `pair ${pair} out of range`);

	const file = resolve(pairDir(dataset, pairId), 'preview', `${layer}.png`);
	if (!existsSync(file)) error(404, `no preview for pair ${pairId} layer ${layer}`);

	const buffer = readFileSync(file);
	return new Response(buffer, {
		headers: { 'Content-Type': 'image/png', 'Cache-Control': 'no-store' }
	});
};
