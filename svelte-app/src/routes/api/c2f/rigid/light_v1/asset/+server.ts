import { error } from '@sveltejs/kit';
import { readFileSync, existsSync } from 'fs';
import { resolve, join } from 'path';
import { pairCount } from '$lib/server/pairs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const ALLOWED = new Set([
	'he.png',
	'ihc.png',
	'ihc_prerot.png',
	'ihc_rigid.png',
	'matches.png',
	'matches.json',
	'result.json',
	'progress.json',
	'field_preview.png',
	'field_fit.json'
]);

export const GET: RequestHandler = ({ url }) => {
	const pair = Number(url.searchParams.get('pair'));
	const name = url.searchParams.get('name') ?? '';
	if (!Number.isInteger(pair) || pair < 0 || pair >= pairCount()) {
		error(400, 'Missing/invalid pair');
	}
	if (!ALLOWED.has(name)) error(400, `asset not allowed: ${name}`);

	const path = join(REPO_ROOT, 'data', 'rigid', 'light_v1', String(pair), 'run', name);
	if (!existsSync(path)) error(404, `missing ${name}`);

	const data = readFileSync(path);
	const isJson = name.endsWith('.json');
	return new Response(data, {
		headers: {
			'Content-Type': isJson ? 'application/json' : 'image/png',
			'Cache-Control': 'no-store'
		}
	});
};
