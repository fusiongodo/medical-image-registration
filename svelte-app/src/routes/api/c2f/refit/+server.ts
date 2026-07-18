import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

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
	const z     = url.searchParams.get('z');
	if (!pair || !depth || (!tau && !z)) error(400, 'Missing pair / depth / (tau or z)');

	const args = [pair, depth, tau ?? '0'];
	if (z) args.push('--z', z);

	try {
		const result = await runRefit(args);
		return json(result);
	} catch (e) {
		error(500, `refit failed: ${(e as Error).message}`);
	}
};
