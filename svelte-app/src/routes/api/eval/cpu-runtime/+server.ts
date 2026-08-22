import { json, error } from '@sveltejs/kit';
import { existsSync, readdirSync, readFileSync } from 'fs';
import { resolve, join } from 'path';
import type { RequestHandler } from './$types';
import { datasetFromUrl, pairCount } from '$lib/server/datasets';

const REPO_ROOT = resolve('..');

function runtimeDir(pairId: number) {
	return resolve(REPO_ROOT, 'data', 'eval_runs', 'cpu_runtime', String(pairId));
}

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const dataset = datasetFromUrl(url);
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount(dataset)) error(404, `pair ${pair} out of range`);

	const dir = runtimeDir(pairId);
	if (!existsSync(dir)) return json({ pair_id: pairId, hosts: {} });

	const hosts: Record<string, unknown> = {};
	for (const name of readdirSync(dir)) {
		if (!name.endsWith('.json')) continue;
		const host = name.slice(0, -5);
		try {
			hosts[host] = JSON.parse(readFileSync(join(dir, name), 'utf-8'));
		} catch {
			/* skip */
		}
	}
	return json({ pair_id: pairId, hosts });
};
