import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { spRotJobs, spRotJobKey, type SpRotJobState } from '$lib/spRotJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'sp_rot_bench_cli.py');

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.run_id !== 'string' || !body.run_id) {
		error(400, 'Expected { run_id: string }');
	}
	const runId = body.run_id as string;
	const key = spRotJobKey(runId);
	const existing = spRotJobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	const state: SpRotJobState = {
		running: true,
		done: 0,
		total: 0,
		detail: 'starting',
		error: null,
		finishedAt: null,
		stage: 'start',
		failed: 0,
		skipped: 0
	};
	spRotJobs.set(key, state);

	const args = ['run', runId];
	if (typeof body.pairs === 'string' && body.pairs) args.push('--pairs', body.pairs);
	if (body.force === true) args.push('--force');

	const child = spawn(PYTHON, [SCRIPT, ...args], { cwd: REPO_ROOT });
	let stdout = '';
	let stderr = '';
	child.stdout.on('data', (chunk: Buffer) => {
		stdout += chunk.toString();
		const lines = stdout.split('\n');
		stdout = lines.pop() ?? '';
		for (const line of lines) {
			const trimmed = line.trim();
			if (!trimmed || trimmed.startsWith('{')) continue;
			const stage = trimmed.match(/\bstage=(\S+)/);
			if (stage) state.stage = stage[1];
			const done = trimmed.match(/\bdone=(\d+)/);
			const total = trimmed.match(/\btotal=(\d+)/);
			const failed = trimmed.match(/\bfailed=(\d+)/);
			const skipped = trimmed.match(/\bskipped=(\d+)/);
			if (done) state.done = parseInt(done[1], 10);
			if (total) state.total = parseInt(total[1], 10);
			if (failed) state.failed = parseInt(failed[1], 10);
			if (skipped) state.skipped = parseInt(skipped[1], 10);
			const pair = trimmed.match(/\bpair=(\S+)/)?.[1];
			const angle = trimmed.match(/\bangle=(\S+)/)?.[1];
			if (pair || angle) {
				state.detail = [pair && `pair ${pair}`, angle && `∠${angle}°`].filter(Boolean).join(' · ');
			}
			spRotJobs.set(key, { ...state });
		}
	});
	child.stderr.on('data', (chunk: Buffer) => {
		const text = chunk.toString();
		stderr += text;
		console.error(`[sp-rot ${runId}] ${text}`);
	});
	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		state.stage = code === 0 ? 'done' : 'error';
		if (code !== 0) {
			const lastLine = stderr.trim().split('\n').filter(Boolean).pop();
			state.error = lastLine || `Process exited with code ${code}`;
		}
		spRotJobs.set(key, { ...state });
	});
	child.on('error', (err) => {
		state.running = false;
		state.error = err.message;
		state.finishedAt = Date.now();
		state.stage = 'error';
		spRotJobs.set(key, { ...state });
	});

	return json({ started: true, state });
};
