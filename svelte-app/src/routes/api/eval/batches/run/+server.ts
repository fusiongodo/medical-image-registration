import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { batchJobs, batchJobKey, type BatchJobState } from '$lib/evalJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'eval_batch_cli.py');

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.batch_id !== 'string' || !body.batch_id) {
		error(400, 'Expected { batch_id: string }');
	}
	const batchId = body.batch_id as string;
	const key = batchJobKey(batchId);
	const existing = batchJobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	const state: BatchJobState = {
		running: true,
		done: 0,
		total: 0,
		detail: 'starting',
		error: null,
		finishedAt: null,
		stage: 'start'
	};
	batchJobs.set(key, state);

	const child = spawn(PYTHON, [SCRIPT, 'run', batchId], { cwd: REPO_ROOT });
	let stdout = '';
	let stderr = '';
	child.stdout.on('data', (chunk: Buffer) => {
		stdout += chunk.toString();
		const lines = stdout.split('\n');
		stdout = lines.pop() ?? '';
		for (const line of lines) {
			const trimmed = line.trim();
			if (!trimmed) continue;
			if (trimmed.startsWith('stage=')) {
				console.log(`[eval batch ${batchId}] ${trimmed}`);
			}
			const stage = trimmed.match(/stage=(\S+)/);
			if (stage) state.stage = stage[1];
			const done = trimmed.match(/\bdone=(\d+)/);
			const total = trimmed.match(/\btotal=(\d+)/);
			if (done) state.done = parseInt(done[1], 10);
			if (total) state.total = parseInt(total[1], 10);
			const detailBits = [
				trimmed.match(/\bpair=(\S+)/)?.[1],
				trimmed.match(/\blam=(\S+)/)?.[1],
				trimmed.match(/\bestimator=(\S+)/)?.[1]
			].filter(Boolean);
			if (detailBits.length) state.detail = detailBits.join(' / ');
			batchJobs.set(key, { ...state });
		}
	});
	child.stderr.on('data', (chunk: Buffer) => {
		const text = chunk.toString();
		stderr += text;
		console.error(`[eval batch ${batchId}] ${text}`);
	});
	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		if (code !== 0) {
			const lastLine = stderr.trim().split('\n').filter(Boolean).pop();
			state.error = lastLine || `Process exited with code ${code}`;
		}
		batchJobs.set(key, { ...state });
	});
	child.on('error', (err) => {
		state.running = false;
		state.error = err.message;
		state.finishedAt = Date.now();
		batchJobs.set(key, { ...state });
	});

	return json({ started: true, state });
};
