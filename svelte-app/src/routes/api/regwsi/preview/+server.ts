import { error } from '@sveltejs/kit';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const LAYERS = new Set(['he', 'ihc_warped']);

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const layer = url.searchParams.get('layer');
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	if (!layer || !LAYERS.has(layer)) error(400, 'layer must be he or ihc_warped');

	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);

	const file = resolve(REPO_ROOT, 'data', 'regwsi', String(pairId), 'preview', `${layer}.png`);
	if (!existsSync(file)) error(404, `no preview for pair ${pairId} layer ${layer}`);

	const buffer = readFileSync(file);
	return new Response(buffer, {
		headers: { 'Content-Type': 'image/png', 'Cache-Control': 'no-store' }
	});
};
