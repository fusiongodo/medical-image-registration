import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

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
			const text = stdout.trim();
			const start = text.indexOf('{');
			const end = text.lastIndexOf('}');
			if (start >= 0 && end >= start) {
				try {
					const obj = JSON.parse(text.slice(start, end + 1)) as {
						ok?: boolean;
						error?: string;
					};
					if (code === 0 || obj.ok === false) return resolveP(obj);
				} catch {
					/* fall through */
				}
			}
			if (code !== 0) {
				const last = stderr.trim().split('\n').filter(Boolean).pop();
				return rejectP(new Error(last || stderr || `exited ${code}`));
			}
			rejectP(new Error('no JSON'));
		});
	});
}

export const GET: RequestHandler = async () => {
	try {
		return json(await runJson(['list']));
	} catch (e) {
		error(500, (e as Error).message);
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.name !== 'string') error(400, 'Expected { name }');
	const args = ['create', '--name', body.name];
	if (typeof body.id === 'string' && body.id) args.push('--id', body.id);
	if (typeof body.pairs === 'string' && body.pairs) args.push('--pairs', body.pairs);
	if (typeof body.batch_size === 'number') args.push('--batch-size', String(body.batch_size));
	if (typeof body.lr === 'number') args.push('--lr', String(body.lr));
	if (typeof body.max_epochs === 'number') args.push('--max-epochs', String(body.max_epochs));
	if (typeof body.ckpt_every_epochs === 'number')
		args.push('--ckpt-every-epochs', String(body.ckpt_every_epochs));
	if (typeof body.eval_every_epochs === 'number')
		args.push('--eval-every-epochs', String(body.eval_every_epochs));
	if (typeof body.log_every === 'number') args.push('--log-every', String(body.log_every));
	if (typeof body.split_seed === 'number') args.push('--split-seed', String(body.split_seed));
	if (typeof body.eval_max_tiles === 'number')
		args.push('--eval-max-tiles', String(body.eval_max_tiles));
	if (body.run_baseline === true) args.push('--run-baseline');
	let result: { ok?: boolean; error?: string };
	try {
		result = (await runJson(args)) as { ok?: boolean; error?: string };
	} catch (e) {
		error(500, (e as Error).message);
	}
	if (result?.ok === false) error(409, result.error || 'create failed');
	return json(result);
};
