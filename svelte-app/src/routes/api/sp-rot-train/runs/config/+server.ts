import { json, error } from '@sveltejs/kit';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');

export const PATCH: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.run_id !== 'string' || typeof body.config !== 'object') {
		error(400, 'Expected { run_id, config }');
	}
	const path = resolve(REPO_ROOT, 'data', 'sp_rot_train', body.run_id, 'config.json');
	if (!existsSync(path)) error(404, 'run not found');
	const cur = JSON.parse(readFileSync(path, 'utf-8'));
	const next = { ...cur, ...body.config, id: cur.id };
	writeFileSync(path, JSON.stringify(next, null, 2));
	return json({ ok: true, config: next });
};
