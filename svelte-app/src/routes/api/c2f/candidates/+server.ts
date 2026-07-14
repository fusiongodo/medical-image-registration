import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import type { RequestHandler } from './$types';
import { jobs, jobKey, type JobState } from '$lib/c2fJobs';

const REPO_ROOT = resolve('..'); // svelte-app sits one level below repo root
const PYTHON    = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT    = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'run.py');
const CACHE_DIR = resolve(REPO_ROOT, 'data', 'c2f_cache');

export const GET: RequestHandler = ({ url }) => {
	const pair  = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const file = join(CACHE_DIR, `${pair}_d${depth}.json`);
	if (!existsSync(file)) return json({ cached: false });

	try {
		const payload = JSON.parse(readFileSync(file, 'utf-8'));
		return json({ cached: true, ...payload });
	} catch {
		return json({ cached: false });
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.pair_id !== 'number' || typeof body.depth !== 'number') {
		error(400, 'Expected { pair_id: number, depth: number }');
	}

	const { pair_id, depth } = body as { pair_id: number; depth: number };
	const key = jobKey(pair_id, depth);

	const existing = jobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	const state: JobState = { running: true, done: 0, total: 0, error: null, finishedAt: null };
	jobs.set(key, state);

	const child = spawn(PYTHON, [SCRIPT, String(pair_id), '--cache-depth', String(depth)], { cwd: REPO_ROOT });

	let stdout = '';
	child.stdout.on('data', (chunk: Buffer) => {
		stdout += chunk.toString();
		for (const line of stdout.split('\n')) {
			const m = line.match(/done=(\d+)\s+total=(\d+)/);
			if (m) {
				state.done = parseInt(m[1], 10);
				state.total = parseInt(m[2], 10);
			}
		}
	});

	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		if (code !== 0) state.error = `Process exited with code ${code}`;
		jobs.set(key, { ...state });
	});

	child.on('error', (err) => {
		state.running = false;
		state.error = err.message;
		state.finishedAt = Date.now();
		jobs.set(key, { ...state });
	});

	return json({ started: true, state });
};
