import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve, join } from 'path';
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync } from 'fs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const ALIGN_SCRIPT = resolve(REPO_ROOT, 'setup', 'auto-alignment', 'align.py');
const METRICS_SCRIPT = resolve(REPO_ROOT, 'setup', 'auto-alignment', 'svelte_metrics.py');
const RUN_SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'run.py');
const CACHE_DIR = resolve(REPO_ROOT, 'data', 'c2f_cache');
const SMOOTH_DIR = resolve(REPO_ROOT, 'data', 'smooth_c2f');
const CROPPED_DIR = resolve(REPO_ROOT, 'data', 'cropped');
const ANNOT_PATH = resolve(REPO_ROOT, 'data', 'registration_annotations.json');

const PREV_COMPLETION_THRESHOLD = 1.0;

interface AlignJobState {
	running: boolean;
	step: string;
	error: string | null;
	finishedAt: number | null;
}

const jobs = new Map<string, AlignJobState>();

function jobKey(pair: number, depth: number): string {
	return `${pair}:${depth}`;
}

function seedStride(depth: number): number {
	return Math.pow(2, depth - 2);
}

function isSeedTile(tile: string, depth: number): boolean {
	const [xi, yi] = tile.split('_').map((n) => parseInt(n, 10));
	const stride = seedStride(depth);
	return Number.isFinite(xi) && Number.isFinite(yi) && xi % stride === 0 && yi % stride === 0;
}

function prevFieldCoversDepth(pair: number, depth: number): boolean {
	if (depth <= 3) return true;
	const fieldPath = join(SMOOTH_DIR, `${pair}_smooth_field.json`);
	if (!existsSync(fieldPath)) return false;
	try {
		const data = JSON.parse(readFileSync(fieldPath, 'utf-8'));
		return data.depths && String(depth - 1) in data.depths;
	} catch {
		return false;
	}
}

function prevSeedCompletion(pair: number, depth: number): number {
	if (depth <= 3) return 1.0;
	const prevDepth = depth - 1;
	const prevDir = join(CROPPED_DIR, String(pair), `d${prevDepth}`);
	if (!existsSync(prevDir)) return 0;

	const seedTiles = readdirSync(prevDir, { withFileTypes: true })
		.filter((e) => e.isDirectory() && e.name.includes('_'))
		.map((e) => e.name)
		.filter((name) => isSeedTile(name, prevDepth));

	if (seedTiles.length === 0) return 0;

	if (!existsSync(ANNOT_PATH)) return 0;
	const all = JSON.parse(readFileSync(ANNOT_PATH, 'utf-8'));
	const entries: { level: number; tile_loc: string }[] = all[String(pair)] || [];
	const done = seedTiles.filter((tile) =>
		entries.some((e) => e.level === prevDepth && e.tile_loc === tile)
	).length;
	return done / seedTiles.length;
}

function canRunAlignment(pair: number, depth: number): { ok: true } | { ok: false; reason: string } {
	if (depth < 3 || depth > 5) {
		return { ok: false, reason: 'Alignment can only be run for levels 3–5' };
	}
	if (!prevFieldCoversDepth(pair, depth)) {
		return { ok: false, reason: `Save the level ${depth - 1} smooth field first` };
	}
	const completion = prevSeedCompletion(pair, depth);
	if (completion < PREV_COMPLETION_THRESHOLD) {
		return {
			ok: false,
			reason: `Complete level ${depth - 1} refine set first (${Math.round(completion * 100)}%)`
		};
	}
	return { ok: true };
}

function spawnPromise(
	script: string,
	args: string[],
	onStep: (step: string) => void,
	stepName: string
): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		onStep(stepName);
		const child = spawn(PYTHON, [script, ...args], { cwd: REPO_ROOT });
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString(); });
		child.stderr.on('data', (chunk: Buffer) => { stderr += chunk.toString(); });
		child.on('close', (code) => {
			if (code !== 0) {
				reject(new Error(`${stepName} failed (code ${code}): ${stderr || stdout}`));
			} else {
				resolve();
			}
		});
		child.on('error', (err) => reject(err));
	});
}

function writeDisplacementsFromCache(pair: number, depth: number) {
	if (depth === 3) return;
	const cachePath = join(CACHE_DIR, `${pair}_d${depth}.json`);
	if (!existsSync(cachePath)) throw new Error('No alignment cache produced');
	const cache = JSON.parse(readFileSync(cachePath, 'utf-8'));
	for (const c of cache.candidates || []) {
		if (typeof c.u !== 'number' || typeof c.v !== 'number' || typeof c.tile_loc !== 'string') continue;
		const elastixDir = join(CROPPED_DIR, String(pair), `d${depth}`, c.tile_loc, 'elastix');
		mkdirSync(elastixDir, { recursive: true });
		writeFileSync(
			join(elastixDir, 'displacement.json'),
			JSON.stringify({ dx: c.u, dy: c.v, psr: c.psr ?? 0 })
		);
	}
}

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.pair_id !== 'number' || typeof body.depth !== 'number') {
		error(400, 'Expected { pair_id: number, depth: number }');
	}
	const { pair_id, depth } = body as { pair_id: number; depth: number };
	const key = jobKey(pair_id, depth);

	const existing = jobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	const gate = canRunAlignment(pair_id, depth);
	if (!gate.ok) {
		return json({ locked: true, reason: gate.reason });
	}

	const state: AlignJobState = { running: true, step: 'starting', error: null, finishedAt: null };
	jobs.set(key, state);

	function setStep(step: string) {
		state.step = step;
		jobs.set(key, { ...state });
	}

	(async () => {
		try {
			if (depth === 3) {
				await spawnPromise(ALIGN_SCRIPT, [String(pair_id), String(depth)], setStep, 'fft alignment');
			} else {
				await spawnPromise(
					RUN_SCRIPT,
					[String(pair_id), '--cache-depth', String(depth)],
					setStep,
					'coarse-to-fine alignment'
				);
				setStep('writing displacements');
				writeDisplacementsFromCache(pair_id, depth);
			}
			await spawnPromise(
				METRICS_SCRIPT,
				[String(pair_id), String(depth)],
				setStep,
				'metrics'
			);
			state.running = false;
			state.step = 'done';
			state.finishedAt = Date.now();
		} catch (err: any) {
			state.running = false;
			state.error = err.message || String(err);
			state.step = 'failed';
			state.finishedAt = Date.now();
		}
		jobs.set(key, { ...state });
	})();

	return json({ started: true, state });
};

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');
	const key = jobKey(parseInt(pair, 10), parseInt(depth, 10));
	const state = jobs.get(key) || null;
	const gate = canRunAlignment(parseInt(pair, 10), parseInt(depth, 10));
	return json({ state, locked: !gate.ok, reason: gate.ok ? null : gate.reason });
};
