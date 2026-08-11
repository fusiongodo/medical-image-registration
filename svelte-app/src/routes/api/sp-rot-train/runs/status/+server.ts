import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { spRotTrainJobs, spRotTrainJobKey } from '$lib/spRotTrainJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'sp_rot_train_cli.py');

function runJson(args: string[]): Promise<unknown> {
	return new Promise((resolveP, rejectP) => {
		const child = spawn(PYTHON, [SCRIPT, ...args], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (c: Buffer) => (stdout += c.toString()));
		child.stderr.on('data', (c: Buffer) => (stderr += c.toString()));
		child.on('error', rejectP);
		child.on('close', (code) => {
			if (code !== 0) {
				return rejectP(new Error(stderr.trim().split('\n').pop() || `exit ${code}`));
			}
			const text = stdout.trim();
			const start = text.indexOf('{');
			const end = text.lastIndexOf('}');
			if (start < 0 || end < start) return rejectP(new Error('no JSON'));
			resolveP(JSON.parse(text.slice(start, end + 1)));
		});
	});
}

function isPidAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

function liveFromPid(runId: string) {
	const p = resolve(REPO_ROOT, 'data', 'sp_rot_train', runId, 'train.pid');
	if (!existsSync(p)) return null;
	const pid = parseInt(readFileSync(p, 'utf-8').trim(), 10);
	if (!Number.isFinite(pid) || !isPidAlive(pid)) return null;
	return { running: true, detail: `pid ${pid}`, error: null, pid };
}

export const GET: RequestHandler = async ({ url }) => {
	const runId = url.searchParams.get('run');
	if (!runId) error(400, 'run required');
	try {
		const disk = (await runJson(['status', runId])) as Record<string, unknown>;
		const mem = spRotTrainJobs.get(spRotTrainJobKey(runId)) ?? null;
		const fromPid = liveFromPid(runId);
		const live =
			mem?.running || fromPid?.running
				? {
						running: true,
						detail: mem?.detail ?? fromPid?.detail ?? null,
						error: null,
						pid: mem?.pid ?? fromPid?.pid
					}
				: (mem ?? fromPid);
		let logs: unknown = null;
		try {
			logs = await runJson(['logs', runId, '-n', '0']);
		} catch {
			logs = null;
		}
		return json({ ...disk, live, logs });
	} catch (e) {
		error(500, (e as Error).message);
	}
};
