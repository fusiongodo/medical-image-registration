import { error } from '@sveltejs/kit';
import { existsSync, readFileSync, mkdirSync, writeFileSync } from 'fs';
import { resolve, normalize, dirname, basename, join } from 'path';
import { spawnSync } from 'child_process';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
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

function thumbPath(srcPath: string, maxW: number): string {
	return join(dirname(srcPath), '.thumbs', `${basename(srcPath)}.w${maxW}.png`);
}

function ensureThumb(srcPath: string, maxW: number): Buffer | null {
	const dest = thumbPath(srcPath, maxW);
	if (existsSync(dest)) return readFileSync(dest);
	mkdirSync(dirname(dest), { recursive: true });
	const py = `
from PIL import Image
import sys
im = Image.open(sys.argv[1])
im.thumbnail((int(sys.argv[2]), int(sys.argv[2])), Image.Resampling.BILINEAR)
im.save(sys.argv[3], format="PNG", optimize=True)
`;
	const r = spawnSync(PYTHON, ['-c', py, srcPath, String(maxW), dest], {
		cwd: REPO_ROOT,
		encoding: 'utf-8',
		maxBuffer: 16 * 1024 * 1024
	});
	if (r.status !== 0 || !existsSync(dest)) return null;
	return readFileSync(dest);
}

export const GET: RequestHandler = ({ url }) => {
	const run = url.searchParams.get('run');
	const pair = url.searchParams.get('pair');
	const angle = url.searchParams.get('angle');
	const name = url.searchParams.get('name');
	const maxWRaw = url.searchParams.get('w');
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

	const isJson = name.endsWith('.json');
	let buf: Buffer = readFileSync(norm);
	const maxW = maxWRaw ? Number(maxWRaw) : NaN;
	if (!isJson && Number.isFinite(maxW) && maxW >= 64 && maxW <= 4096) {
		const resized = ensureThumb(norm, Math.floor(maxW));
		if (resized) buf = resized;
	}

	return new Response(buf, {
		headers: {
			'Content-Type': isJson ? 'application/json' : 'image/png',
			'Cache-Control': isJson
				? 'private, max-age=60'
				: 'private, max-age=86400, immutable'
		}
	});
};
