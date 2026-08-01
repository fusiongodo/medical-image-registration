import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { pairCount } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'regWSI', 'tre_cli.py');

function runTre(pairId: number): Promise<unknown> {
	return new Promise((resolveP, rejectP) => {
		const child = spawn(PYTHON, [SCRIPT, String(pairId)], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (c: Buffer) => (stdout += c.toString()));
		child.stderr.on('data', (c: Buffer) => (stderr += c.toString()));
		child.on('error', (err) => rejectP(err));
		child.on('close', (code) => {
			if (code !== 0) return rejectP(new Error(stderr || `exited ${code}`));
			try {
				resolveP(JSON.parse(stdout.trim()));
			} catch (e) {
				rejectP(new Error(`bad tre output: ${(e as Error).message}`));
			}
		});
	});
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);
	try {
		const result = await runTre(pairId);
		return json(result);
	} catch (e) {
		error(500, `tre failed: ${(e as Error).message}`);
	}
};
