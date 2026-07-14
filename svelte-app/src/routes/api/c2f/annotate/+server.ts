import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON    = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT    = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'annotate_cli.py');
const STORE     = join(REPO_ROOT, 'data', 'registration_annotations.json');

interface Entry { level: number; tile_loc: string; type: string; disp: { u: number; v: number }; }

export const GET: RequestHandler = ({ url }) => {
	const pair  = url.searchParams.get('pair');
	const level = url.searchParams.get('level');
	if (!pair || !level) error(400, 'Missing pair / level');
	if (!existsSync(STORE)) return json({});

	try {
		const store = JSON.parse(readFileSync(STORE, 'utf-8')) as Record<string, Entry[]>;
		const lvl = parseInt(level, 10);
		const out: Record<string, { type: string; u: number; v: number }> = {};
		for (const e of store[pair] ?? []) {
			if (e.level === lvl) out[e.tile_loc] = { type: e.type, u: e.disp.u, v: e.disp.v };
		}
		return json(out);
	} catch {
		return json({});
	}
};

function runAnnotate(args: string[]): Promise<unknown> {
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
				rejectP(new Error(`bad annotate output: ${(e as Error).message}`));
			}
		});
	});
}

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (
		!body ||
		typeof body.pair_id !== 'number' ||
		typeof body.level !== 'number' ||
		typeof body.tile_loc !== 'string' ||
		!['approve', 'correct', 'exclude', 'clear'].includes(body.action)
	) {
		error(400, 'Expected { pair_id, level, tile_loc, action: "approve"|"correct"|"exclude"|"clear", u?, v? }');
	}

	const { pair_id, level, tile_loc, action } = body as {
		pair_id: number;
		level: number;
		tile_loc: string;
		action: 'approve' | 'correct' | 'exclude' | 'clear';
	};

	const args = [String(pair_id), String(level), tile_loc, action];
	if (action !== 'clear') {
		if (action !== 'exclude' && (typeof body.u !== 'number' || typeof body.v !== 'number')) {
			error(400, `${action} requires numeric u, v`);
		}
		if (typeof body.u === 'number' && typeof body.v === 'number') {
			args.push(String(body.u), String(body.v));
		}
	}

	try {
		const result = await runAnnotate(args);
		return json(result);
	} catch (e) {
		error(500, `annotate failed: ${(e as Error).message}`);
	}
};
