import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { normalizeDataset, pairCount } from '$lib/server/datasets';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'eval_batch_cli.py');

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
				rejectP(new Error(`bad batches output: ${(e as Error).message}`));
			}
		});
	});
}

export const GET: RequestHandler = async () => {
	try {
		const result = (await runJson(['list'])) as { batches?: unknown[] };
		return json(result);
	} catch (e) {
		error(500, `list batches failed: ${(e as Error).message}`);
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.name !== 'string' || !Array.isArray(body.pairs)) {
		error(400, 'Expected { name: string, pairs: number[], ... }');
	}
	const dataset = normalizeDataset(typeof body.dataset === 'string' ? body.dataset : 'muromi');
	const n = pairCount(dataset);
	const pairs = (body.pairs as unknown[])
		.map((p) => Number(p))
		.filter((p) => Number.isInteger(p) && p >= 0 && p < n);
	if (!pairs.length) error(400, 'No valid pairs');

	const args = ['create', '--name', body.name, '--pairs', pairs.join(','), '--dataset', dataset];
	if (typeof body.id === 'string' && body.id) args.push('--id', body.id);
	if (typeof body.notes === 'string') args.push('--notes', body.notes);
	if (Array.isArray(body.lams) && body.lams.length) args.push('--lams', body.lams.join(','));
	if (Array.isArray(body.estimators) && body.estimators.length) {
		args.push('--estimators', body.estimators.join(','));
	}
	const cfg = body.config ?? {};
	if (typeof cfg.wendland_eps === 'number') args.push('--wendland-eps', String(cfg.wendland_eps));
	if (typeof cfg.bspline_grid === 'number') args.push('--bspline-grid', String(cfg.bspline_grid));
	if (typeof cfg.bspline_reg === 'number') args.push('--bspline-reg', String(cfg.bspline_reg));
	if (cfg.force === true) args.push('--force');

	try {
		const result = (await runJson(args)) as { ok?: boolean; error?: string };
		if (result && result.ok === false) {
			error(409, result.error || 'create failed');
		}
		return json(result);
	} catch (e) {
		const msg = (e as Error).message;
		const status = /already exists/i.test(msg) ? 409 : 500;
		error(status, `create batch failed: ${msg}`);
	}
};
