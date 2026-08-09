import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'sp_rot_bench_cli.py');
const LABELS = new Set(['pass', 'fail', 'unsure']);

function runJson(args: string[]): Promise<unknown> {
	return new Promise((resolveP, rejectP) => {
		const child = spawn(PYTHON, [SCRIPT, ...args], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (c: Buffer) => (stdout += c.toString()));
		child.stderr.on('data', (c: Buffer) => (stderr += c.toString()));
		child.on('error', (err) => rejectP(err));
		child.on('close', (code) => {
			if (code !== 0) {
				const last = stderr.trim().split('\n').filter(Boolean).pop();
				return rejectP(new Error(last || stderr || `exited ${code}`));
			}
			try {
				const text = stdout.trim();
				const start = text.indexOf('{');
				const end = text.lastIndexOf('}');
				if (start < 0 || end < start) throw new Error('no JSON object in output');
				resolveP(JSON.parse(text.slice(start, end + 1)));
			} catch (e) {
				rejectP(new Error(`bad label output: ${(e as Error).message}`));
			}
		});
	});
}

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (
		!body ||
		typeof body.run_id !== 'string' ||
		typeof body.pair !== 'number' ||
		typeof body.angle !== 'number' ||
		typeof body.label !== 'string'
	) {
		error(400, 'Expected { run_id, pair, angle, label }');
	}
	const label = body.label.toLowerCase();
	if (!LABELS.has(label)) error(400, 'label must be pass|fail|unsure');

	const args = [
		'label',
		body.run_id,
		'--pair',
		String(body.pair),
		'--angle',
		String(body.angle),
		'--label',
		label
	];
	if (typeof body.note === 'string' && body.note) args.push('--note', body.note);

	try {
		const result = (await runJson(args)) as { ok?: boolean; error?: string };
		if (result && result.ok === false) error(400, result.error || 'label failed');
		return json(result);
	} catch (e) {
		error(500, `label failed: ${(e as Error).message}`);
	}
};
