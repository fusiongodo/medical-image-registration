import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import { pairCount } from '$lib/server/pairs';
import { jobs, rigidJobKey, type JobState } from '$lib/c2fJobs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'rigid_cli.py');

function validPair(pair: unknown): pair is number {
	return typeof pair === 'number' && Number.isInteger(pair) && pair >= 0 && pair < pairCount();
}

function runCli(args: string[]): Promise<unknown> {
	return new Promise((resolveP, rejectP) => {
		const child = spawn(PYTHON, [SCRIPT, ...args], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (c: Buffer) => (stdout += c.toString()));
		child.stderr.on('data', (c: Buffer) => (stderr += c.toString()));
		child.on('error', (err) => rejectP(err));
		child.on('close', (code) => {
			if (code !== 0) return rejectP(new Error(stderr || `exited ${code}`));
			try {
				resolveP(JSON.parse(stdout.trim().split('\n').pop() ?? '{}'));
			} catch (e) {
				rejectP(new Error(`bad rigid output: ${(e as Error).message}\n${stdout}`));
			}
		});
		child.stdin.end();
	});
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = Number(url.searchParams.get('pair'));
	if (!validPair(pair)) error(400, 'Missing/invalid pair');
	try {
		return json(await runCli(['get', String(pair)]));
	} catch (e) {
		error(500, `rigid get failed: ${(e as Error).message}`);
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || !validPair(body.pair_id)) {
		error(400, 'Expected { pair_id: number, action: run|save|clear, ... }');
	}

	const action = body.action as string;
	const pair_id = body.pair_id as number;

	if (action === 'save') {
		try {
			return json(await runCli(['save', String(pair_id)]));
		} catch (e) {
			error(500, `rigid save failed: ${(e as Error).message}`);
		}
	}

	if (action === 'clear') {
		const args = ['clear', String(pair_id)];
		if (body.clear_run) args.push('--run');
		try {
			return json(await runCli(args));
		} catch (e) {
			error(500, `rigid clear failed: ${(e as Error).message}`);
		}
	}

	if (action === 'reclassify') {
		if (typeof body.inlier_px !== 'number') {
			error(400, 'reclassify expects { inlier_px: number }');
		}
		try {
			const result = (await runCli([
				'reclassify',
				String(pair_id),
				'--inlier-px',
				String(body.inlier_px)
			])) as { error?: string };
			if (result.error) error(400, result.error);
			return json(result);
		} catch (e) {
			error(500, `reclassify failed: ${(e as Error).message}`);
		}
	}

	if (action === 'field-fit') {
		const args = [
			'field-fit',
			String(pair_id),
			'--estimator',
			typeof body.field_estimator === 'string' ? body.field_estimator : 'tps'
		];
		if (typeof body.wendland_epsilon === 'number') {
			args.push('--wendland-eps', String(body.wendland_epsilon));
		}
		if (typeof body.bspline_grid === 'number') {
			args.push('--bspline-grid', String(body.bspline_grid));
		}
		if (typeof body.bspline_reg === 'number') {
			args.push('--bspline-reg', String(body.bspline_reg));
		}
		if (body.all_matches) args.push('--all-matches');
		try {
			const result = (await runCli(args)) as { error?: string };
			if (result.error) error(400, result.error);
			return json(result);
		} catch (e) {
			error(500, `field-fit failed: ${(e as Error).message}`);
		}
	}

	if (action === 'run') {
		const key = rigidJobKey(pair_id);
		const existing = jobs.get(key);
		if (existing?.running) {
			return json({ started: false, state: existing });
		}

		const state: JobState = {
			running: true,
			done: 0,
			total: 7,
			error: null,
			finishedAt: null,
			stage: 'starting'
		};
		jobs.set(key, state);

		const level = typeof body.preview_level === 'number' ? body.preview_level : 2;
		const preRot = typeof body.pre_rotation_deg === 'number' ? body.pre_rotation_deg : 0;
		const hp =
			body.hyperparams && typeof body.hyperparams === 'object'
				? JSON.stringify(body.hyperparams)
				: '';

		const args = [
			'run',
			String(pair_id),
			'--level',
			String(level),
			'--pre-rot',
			String(preRot)
		];
		if (hp) args.push('--hyperparams', hp);

		const child = spawn(PYTHON, [SCRIPT, ...args], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		const stages = ['load', 'prerot', 'superpoint', 'lightglue', 'fit', 'preview', 'done'];

		child.stdout.on('data', (chunk: Buffer) => {
			stdout += chunk.toString();
			for (const line of stdout.split('\n')) {
				const m = line.match(/^stage=(\S+)/);
				if (m) {
					state.stage = m[1];
					const idx = stages.indexOf(m[1]);
					if (idx >= 0) state.done = idx + 1;
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
				state.stage = 'error';
			} else {
				state.stage = 'done';
				state.done = state.total;
				try {
					const result = JSON.parse(stdout.trim().split('\n').pop() ?? '{}') as {
						error?: string;
					};
					if (result.error) {
						state.error = result.error;
						state.stage = 'error';
					}
				} catch {
					/* ignore parse for job state */
				}
			}
			jobs.set(key, { ...state });
		});
		child.on('error', (err) => {
			state.running = false;
			state.error = err.message;
			state.stage = 'error';
			state.finishedAt = Date.now();
			jobs.set(key, { ...state });
		});

		return json({ started: true, state });
	}

	error(400, `unknown action ${action}`);
};
