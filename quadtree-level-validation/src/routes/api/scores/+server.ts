import { json, error } from '@sveltejs/kit';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const SCORES_PATH = resolve('..', 'data', 'lncc_scores.json');

type TileScores = { lncc?: number; sq?: number };
type ScoreStore = Record<string, Record<string, Record<string, Record<string, TileScores>>>>;

function read(): ScoreStore {
	if (!existsSync(SCORES_PATH)) return {};
	try { return JSON.parse(readFileSync(SCORES_PATH, 'utf-8')); }
	catch { return {}; }
}

function write(store: ScoreStore) {
	writeFileSync(SCORES_PATH, JSON.stringify(store), 'utf-8');
}

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	const patchSize = url.searchParams.get('patchSize');
	if (!pair || !depth || !patchSize) error(400, 'Missing pair / depth / patchSize');

	const store = read();
	const tiles = store[pair]?.[depth]?.[patchSize] ?? {};
	return json(tiles);
};

export const POST: RequestHandler = async ({ request }) => {
	const { pair_id, depth, patchSize, entries } = await request.json();
	if (typeof pair_id !== 'number' || typeof depth !== 'number' || typeof patchSize !== 'number') {
		error(400, 'Expected { pair_id, depth, patchSize: number, entries: object }');
	}

	const store = read();
	const p = String(pair_id), d = String(depth), ps = String(patchSize);
	store[p] ??= {};
	store[p][d] ??= {};
	store[p][d][ps] ??= {};

	for (const [tile, scores] of Object.entries(entries as Record<string, TileScores>)) {
		store[p][d][ps][tile] = { ...store[p][d][ps][tile], ...scores };
	}

	write(store);
	return json({ ok: true });
};
