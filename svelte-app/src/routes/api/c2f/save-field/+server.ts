import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'refit_cli.py');

const LAMS = new Set(['fft', 'superpoint_glue']);
const ESTIMATORS = new Set(['tps', 'wendland', 'bspline']);

function normalizeLam(raw: unknown): string {
	return typeof raw === 'string' && LAMS.has(raw) ? raw : 'fft';
}

function normalizeEstimator(raw: unknown): string {
	return typeof raw === 'string' && ESTIMATORS.has(raw) ? raw : 'tps';
}

function runSave(args: string[]): Promise<unknown> {
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
				rejectP(new Error(`bad save output: ${(e as Error).message}`));
			}
		});
	});
}

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	const hasKeep = body && typeof body.keep === 'number';
	if (
		!body ||
		typeof body.pair_id !== 'number' ||
		typeof body.depth !== 'number' ||
		(typeof body.tau !== 'number' && !hasKeep)
	) {
		error(400, 'Expected { pair_id: number, depth: number, tau?: number, keep?: number, lam? }');
	}

	const { pair_id, depth, tau, keep } = body as {
		pair_id: number;
		depth: number;
		tau?: number;
		keep?: number;
		field_estimator?: string;
		lam?: string;
	};
	const field_estimator = normalizeEstimator(body.field_estimator);
	const lam = normalizeLam(body.lam);

	const args = [
		String(pair_id),
		String(depth),
		String(tau ?? 0),
		'--save',
		'--lam',
		lam,
		'--field-estimator',
		field_estimator
	];
	if (hasKeep) args.push('--keep', String(keep));

	try {
		const result = await runSave(args);
		return json(result);
	} catch (e) {
		error(500, `save failed: ${(e as Error).message}`);
	}
};
