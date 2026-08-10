import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { spRotTrainJobs, spRotTrainJobKey, type SpRotTrainJobState } from '$lib/spRotTrainJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'sp_rot_train_cli.py');

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.run_id !== 'string' || typeof body.cmd !== 'string') {
		error(400, 'Expected { run_id, cmd }');
	}
	const runId = body.run_id as string;
	const cmd = body.cmd as string;
	if (!['run', 'resume', 'pause', 'stop'].includes(cmd)) error(400, 'bad cmd');

	if (cmd === 'pause' || cmd === 'stop') {
		const child = spawn(PYTHON, [SCRIPT, cmd, runId], { cwd: REPO_ROOT });
		await new Promise<void>((res) => child.on('close', () => res()));
		return json({ ok: true, cmd });
	}

	const key = spRotTrainJobKey(runId);
	const existing = spRotTrainJobs.get(key);
	if (existing?.running) return json({ started: false, state: existing });

	const state: SpRotTrainJobState = {
		running: true,
		detail: cmd,
		error: null,
		finishedAt: null,
		cmd
	};
	spRotTrainJobs.set(key, state);

	const child = spawn(PYTHON, [SCRIPT, cmd, runId], { cwd: REPO_ROOT });
	let stderr = '';
	child.stdout.on('data', (chunk: Buffer) => {
		const text = chunk.toString();
		const line = text.trim().split('\n').filter(Boolean).pop();
		if (line && !line.startsWith('{')) state.detail = line.slice(0, 200);
		spRotTrainJobs.set(key, { ...state });
	});
	child.stderr.on('data', (chunk: Buffer) => {
		stderr += chunk.toString();
	});
	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		if (code !== 0) {
			state.error = stderr.trim().split('\n').filter(Boolean).pop() || `exit ${code}`;
		}
		spRotTrainJobs.set(key, { ...state });
	});
	return json({ started: true, state });
};
