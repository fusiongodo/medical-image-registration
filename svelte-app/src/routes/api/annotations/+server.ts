import { json, error } from '@sveltejs/kit';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const ANNOTATIONS_PATH = resolve('..', 'data', 'point_annotations.json');

interface Point { x: number; y: number; }
interface TileAnnotation { hePoints: Point[]; ihcPoints: Point[]; }
type AnnotationStore = Record<string, Record<string, Record<string, TileAnnotation>>>;

function read(): AnnotationStore {
	if (!existsSync(ANNOTATIONS_PATH)) return {};
	try { return JSON.parse(readFileSync(ANNOTATIONS_PATH, 'utf-8')); }
	catch { return {}; }
}

function write(store: AnnotationStore) {
	writeFileSync(ANNOTATIONS_PATH, JSON.stringify(store, null, 2), 'utf-8');
}

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');
	const store = read();
	return json(store[pair]?.[depth] ?? {});
};

export const POST: RequestHandler = async ({ request }) => {
	const { pair_id, depth, tile, hePoints, ihcPoints } = await request.json();
	if (typeof pair_id !== 'number' || typeof depth !== 'number' || typeof tile !== 'string') {
		error(400, 'Expected { pair_id, depth: number, tile: string, hePoints, ihcPoints }');
	}
	const store = read();
	const p = String(pair_id), d = String(depth);
	store[p] ??= {};
	store[p][d] ??= {};
	store[p][d][tile] = { hePoints, ihcPoints };
	write(store);
	return json({ ok: true });
};
