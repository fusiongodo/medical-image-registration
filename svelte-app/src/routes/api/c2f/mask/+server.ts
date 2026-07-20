import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON    = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT    = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'mask_cli.py');
const STORE     = join(REPO_ROOT, 'data', 'masked_out.json');

interface MaskEntry { level: number; tile_loc: string; state?: 'mask' | 'unmask'; }

// Effective masked state for one tile: the deepest-level explicit entry among
// the tile and its ancestors decides (masks propagate forward by quadtree
// index). Mirrors setup/coarse_to_fine/masks.py::_resolve.
function isMasked(entries: MaskEntry[], level: number, x: number, y: number): boolean {
	let bestLevel = -1;
	let bestState: 'mask' | 'unmask' | null = null;
	for (const e of entries) {
		const el = e.level;
		if (el > level) continue;
		const [ex, ey] = e.tile_loc.split('_').map((n) => parseInt(n, 10));
		const shift = level - el;
		if (x >> shift === ex && y >> shift === ey && el > bestLevel) {
			bestLevel = el;
			bestState = e.state ?? 'mask';
		}
	}
	return bestState === 'mask';
}

export const GET: RequestHandler = ({ url }) => {
	const pair  = url.searchParams.get('pair');
	const level = url.searchParams.get('level');
	if (!pair || !level) error(400, 'Missing pair / level');
	if (!existsSync(STORE)) return json({});

	try {
		const store = JSON.parse(readFileSync(STORE, 'utf-8')) as Record<string, MaskEntry[]>;
		const entries = store[pair] ?? [];
		const lvl = parseInt(level, 10);
		const grid = 2 ** lvl;
		const out: Record<string, true> = {};
		for (let y = 0; y < grid; y++) {
			for (let x = 0; x < grid; x++) {
				if (isMasked(entries, lvl, x, y)) out[`${x}_${y}`] = true;
			}
		}
		return json(out);
	} catch {
		return json({});
	}
};

function runMask(args: string[]): Promise<unknown> {
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
				rejectP(new Error(`bad mask output: ${(e as Error).message}`));
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
		!['mask', 'unmask', 'clear'].includes(body.action)
	) {
		error(400, 'Expected { pair_id, level, tile_loc, action: "mask"|"unmask"|"clear" }');
	}

	const { pair_id, level, tile_loc, action } = body as {
		pair_id: number;
		level: number;
		tile_loc: string;
		action: 'mask' | 'unmask' | 'clear';
	};

	try {
		const result = await runMask([String(pair_id), String(level), tile_loc, action]);
		return json(result);
	} catch (e) {
		error(500, `mask failed: ${(e as Error).message}`);
	}
};
