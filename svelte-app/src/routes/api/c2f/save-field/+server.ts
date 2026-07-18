import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON    = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT    = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'refit_cli.py');

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
	const hasZ = body && typeof body.z === 'number';
	if (
		!body ||
		typeof body.pair_id !== 'number' ||
		typeof body.depth !== 'number' ||
		(typeof body.tau !== 'number' && !hasZ)
	) {
		error(400, 'Expected { pair_id: number, depth: number, tau?: number, z?: number }');
	}

	const { pair_id, depth, tau, z } = body as {
		pair_id: number;
		depth: number;
		tau?: number;
		z?: number;
	};

	const args = [String(pair_id), String(depth), String(tau ?? 0), '--save'];
	if (hasZ) args.push('--z', String(z));

	try {
		const result = await runSave(args);
		return json(result);
	} catch (e) {
		error(500, `save failed: ${(e as Error).message}`);
	}
};
