import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const PYTHON    = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT    = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'refit_cli.py');

function runRefit(args: string[]): Promise<unknown> {
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
				rejectP(new Error(`bad refit output: ${(e as Error).message}`));
			}
		});
	});
}

export const GET: RequestHandler = async ({ url }) => {
	const pair  = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	const tau   = url.searchParams.get('tau');
	const keep  = url.searchParams.get('keep');
	const fieldEstimator = url.searchParams.get('field_estimator') ?? 'tps';
	if (!pair || !depth || (!tau && !keep)) error(400, 'Missing pair / depth / (tau or keep)');

	const pairId = parseInt(pair, 10);
	if (isNaN(pairId) || pairId < 0 || pairId >= pairCount()) {
		error(400, `Pair ${pair} does not exist (valid range 0..${pairCount() - 1})`);
	}

	const args = [pair, depth, tau ?? '0', '--field-estimator', fieldEstimator];
	if (keep) args.push('--keep', keep);

	try {
		const result = await runRefit(args);
		return json(result);
	} catch (e) {
		error(500, `refit failed: ${(e as Error).message}`);
	}
};
