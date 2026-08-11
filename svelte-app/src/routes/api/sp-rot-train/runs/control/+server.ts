import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { openSync, writeFileSync, mkdirSync, existsSync, readFileSync, unlinkSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { spRotTrainJobs, spRotTrainJobKey, type SpRotTrainJobState } from '$lib/spRotTrainJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'sp_rot_train_cli.py');

function runDir(runId: string) {
	return resolve(REPO_ROOT, 'data', 'sp_rot_train', runId);
}

function pidPath(runId: string) {
	return resolve(runDir(runId), 'train.pid');
}

function isPidAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

function readPid(runId: string): number | null {
	const p = pidPath(runId);
	if (!existsSync(p)) return null;
	const n = parseInt(readFileSync(p, 'utf-8').trim(), 10);
	return Number.isFinite(n) ? n : null;
}

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
	const alivePid = readPid(runId);
	if (existing?.running || (alivePid != null && isPidAlive(alivePid))) {
		return json({
			started: false,
			state: existing ?? { running: true, detail: 'already running', error: null, finishedAt: null, cmd }
		});
	}

	const logsDir = resolve(runDir(runId), 'logs');
	mkdirSync(logsDir, { recursive: true });
	const outFd = openSync(resolve(logsDir, 'train.out'), 'a');
	const errFd = openSync(resolve(logsDir, 'train.err'), 'a');

	const state: SpRotTrainJobState = {
		running: true,
		detail: cmd,
		error: null,
		finishedAt: null,
		cmd
	};
	spRotTrainJobs.set(key, state);

	const child = spawn(PYTHON, ['-u', SCRIPT, cmd, runId], {
		cwd: REPO_ROOT,
		detached: true,
		stdio: ['ignore', outFd, errFd]
	});
	if (child.pid != null) {
		writeFileSync(pidPath(runId), String(child.pid));
		state.pid = child.pid;
		spRotTrainJobs.set(key, { ...state });
	}
	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		if (code !== 0 && code != null) {
			try {
				const errText = existsSync(resolve(logsDir, 'train.err'))
					? readFileSync(resolve(logsDir, 'train.err'), 'utf-8')
					: '';
				state.error = errText.trim().split('\n').filter(Boolean).pop() || `exit ${code}`;
			} catch {
				state.error = `exit ${code}`;
			}
		}
		try {
			unlinkSync(pidPath(runId));
		} catch {
			/* ignore */
		}
		spRotTrainJobs.set(key, { ...state });
	});
	child.unref();

	return json({ started: true, state });
};
