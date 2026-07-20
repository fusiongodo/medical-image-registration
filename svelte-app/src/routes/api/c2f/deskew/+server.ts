import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve } from 'path';
import { pairCount } from '$lib/server/pairs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'deskew_cli.py');

/** Spawn deskew_cli.py; `stdin` (points JSON) is written to the child when given. */
function runDeskew(args: string[], stdin?: string): Promise<unknown> {
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
				rejectP(new Error(`bad deskew output: ${(e as Error).message}`));
			}
		});
		if (stdin !== undefined) {
			child.stdin.write(stdin);
		}
		child.stdin.end();
	});
}

function validPair(pair: unknown): pair is number {
	return typeof pair === 'number' && Number.isInteger(pair) && pair >= 0 && pair < pairCount();
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = Number(url.searchParams.get('pair'));
	if (!validPair(pair)) error(400, 'Missing/invalid pair');
	try {
		return json(await runDeskew(['get', String(pair)]));
	} catch (e) {
		error(500, `deskew get failed: ${(e as Error).message}`);
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || !validPair(body.pair_id)) {
		error(400, 'Expected { pair_id: number, depth?, points?, action? }');
	}

	const action = body.action ?? 'apply';
	if (action !== 'apply' && action !== 'clear') error(400, `unknown action ${action}`);
	if (action === 'apply' && (typeof body.depth !== 'number' || !Array.isArray(body.points))) {
		error(400, 'apply expects { depth: number, points: [{he,ihc}] }');
	}

	const args =
		action === 'clear'
			? ['clear', String(body.pair_id)]
			: ['apply', String(body.pair_id), String(body.depth)];
	const stdin = action === 'apply' ? JSON.stringify({ points: body.points }) : undefined;

	try {
		return json(await runDeskew(args, stdin));
	} catch (e) {
		error(500, `deskew failed: ${(e as Error).message}`);
	}
};
