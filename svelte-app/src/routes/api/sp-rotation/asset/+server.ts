import { error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve, normalize } from 'path';
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
	'gt_rigid.json'
]);

export const GET: RequestHandler = ({ url }) => {
	const run = url.searchParams.get('run');
	const pair = url.searchParams.get('pair');
	const angle = url.searchParams.get('angle');
	const name = url.searchParams.get('name');
	if (!run || !pair || !name) error(400, 'Missing run, pair, or name');
	if (!ALLOWED.has(name)) error(400, 'Invalid asset name');

	const root = resolve(REPO_ROOT, 'data', 'sp_rot_runs');
	let path: string;
	if (name === 'gt_rigid.json') {
		path = resolve(root, run, pair, name);
	} else {
		if (angle == null) error(400, 'Missing angle');
		path = resolve(root, run, pair, angle, name);
	}
	const norm = normalize(path);
	if (!norm.startsWith(root)) error(400, 'Invalid path');
	if (!existsSync(norm)) error(404, 'Not found');

	const buf = readFileSync(norm);
	const isJson = name.endsWith('.json');
	return new Response(buf, {
		headers: {
			'Content-Type': isJson ? 'application/json' : 'image/png',
			'Cache-Control': 'no-store'
		}
	});
};
