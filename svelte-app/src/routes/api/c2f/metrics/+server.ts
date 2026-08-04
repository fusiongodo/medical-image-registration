import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { jobs, jobKey, type JobState } from '$lib/c2fJobs';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'run.py');

const LAMS = new Set(['fft', 'superpoint_glue']);

function normalizeLam(raw: unknown): string {
	return typeof raw === 'string' && LAMS.has(raw) ? raw : 'fft';
}

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.pair_id !== 'number' || typeof body.depth !== 'number') {
		error(400, 'Expected { pair_id: number, depth: number, lam? }');
	}

	const { pair_id, depth } = body as { pair_id: number; depth: number; lam?: string };
	const lam = normalizeLam(body.lam);
	if (pair_id < 0 || pair_id >= pairCount()) {
		error(400, `Pair ${pair_id} does not exist (valid range 0..${pairCount() - 1})`);
	}
	const key = jobKey(pair_id, depth, 'metrics', lam);

	const existing = jobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	const state: JobState = { running: true, done: 0, total: 0, error: null, finishedAt: null };
	jobs.set(key, state);

	const child = spawn(
		PYTHON,
		[SCRIPT, String(pair_id), '--metrics-depth', String(depth), '--lam', lam],
		{ cwd: REPO_ROOT }
	);

	let stdout = '';
	let stderr = '';
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
	child.stderr.on('data', (chunk: Buffer) => {
		stderr += chunk.toString();
	});

	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		if (code !== 0) {
			const lastLine = stderr.trim().split('\n').filter(Boolean).pop();
			state.error = lastLine || `Process exited with code ${code}`;
		}
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
