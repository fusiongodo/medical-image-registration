import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'field_set_cli.py');

function run(args: string[]): Promise<unknown> {
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
				rejectP(new Error(`bad field-set output: ${(e as Error).message}`));
			}
		});
	});
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	if (!pair) error(400, 'Missing pair');
	try {
		return json(await run(['list', pair]));
	} catch (e) {
		error(500, `field-set list failed: ${(e as Error).message}`);
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	const valid = ['save', 'load', 'new', 'delete', 'rename'];
	if (!body || typeof body.pair_id !== 'number' || !valid.includes(body.action)) {
		error(400, 'Expected { pair_id: number, action: save|load|new|delete|rename, set_id?, name? }');
	}

	const { pair_id, action, set_id, name } = body as {
		pair_id: number;
		action: 'save' | 'load' | 'new' | 'delete' | 'rename';
		set_id?: string;
		name?: string;
	};

	const args: string[] = [action, String(pair_id)];
	if (action === 'save') {
		if (typeof name !== 'string' || !name.trim()) error(400, 'save requires name');
		args.push('--name', name.trim());
		if (set_id) args.push('--id', set_id);
	} else if (action === 'new') {
		if (typeof name !== 'string' || !name.trim()) error(400, 'new requires name');
		args.push('--name', name.trim());
	} else if (action === 'load' || action === 'delete') {
		if (!set_id) error(400, `${action} requires set_id`);
		args.push('--id', set_id);
	} else if (action === 'rename') {
		if (!set_id) error(400, 'rename requires set_id');
		if (typeof name !== 'string' || !name.trim()) error(400, 'rename requires name');
		args.push('--id', set_id, '--name', name.trim());
	}

	try {
		return json(await run(args));
	} catch (e) {
		error(500, `field-set ${action} failed: ${(e as Error).message}`);
	}
};
