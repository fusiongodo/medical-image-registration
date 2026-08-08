import { json, error } from '@sveltejs/kit';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { resolve, dirname } from 'path';
import type { RequestHandler } from './$types';
import { pairCount, pairFingerprint, fingerprintMatches } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');

function landmarksPath(pairId: number) {
	return resolve(REPO_ROOT, 'data', 'regwsi', String(pairId), 'landmarks.json');
}

function empty(pairId: number) {
	return {
		pair_id: pairId,
		identity: pairFingerprint(pairId) ?? undefined,
		points: [] as { he: [number, number]; ihc: [number, number] }[]
	};
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);

	const path = landmarksPath(pairId);
	if (!existsSync(path)) return json(empty(pairId));
	try {
		const data = JSON.parse(readFileSync(path, 'utf-8'));
		if (!fingerprintMatches(pairId, data.identity)) {
			return json({ ...empty(pairId), stale: true });
		}
		return json(data);
	} catch {
		error(500, 'failed to read landmarks');
	}
};

export const POST: RequestHandler = async ({ request, url }) => {
	const pair = url.searchParams.get('pair');
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);

	let body: { points?: { he: [number, number]; ihc: [number, number] }[] };
	try {
		body = await request.json();
	} catch {
		error(400, 'invalid JSON');
	}
	const points = Array.isArray(body.points) ? body.points : [];
	for (const p of points) {
		if (
			!p?.he ||
			!p?.ihc ||
			p.he.length !== 2 ||
			p.ihc.length !== 2 ||
			p.he.some((v) => typeof v !== 'number' || v < 0 || v > 1) ||
			p.ihc.some((v) => typeof v !== 'number' || v < 0 || v > 1)
		) {
			error(400, 'points must be {he:[x,y], ihc:[x,y]} in [0,1]');
		}
	}

	const data = {
		pair_id: pairId,
		identity: pairFingerprint(pairId),
		points
	};
	const path = landmarksPath(pairId);
	mkdirSync(dirname(path), { recursive: true });
	writeFileSync(path, JSON.stringify(data));
	return json(data);
};

export const DELETE: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);

	const data = empty(pairId);
	const path = landmarksPath(pairId);
	mkdirSync(dirname(path), { recursive: true });
	writeFileSync(path, JSON.stringify(data));
	return json(data);
};
