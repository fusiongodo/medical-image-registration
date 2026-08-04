import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { resolve, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import type { RequestHandler } from './$types';
import { jobs, jobKey, type JobState } from '$lib/c2fJobs';
import { pairCount, fingerprintMatches } from '$lib/server/pairs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'run.py');
const CACHE_ROOT = resolve(REPO_ROOT, 'data', 'c2f_cache');

const LAMS = new Set(['fft', 'superpoint_glue']);

function normalizeLam(raw: unknown): string {
	return typeof raw === 'string' && LAMS.has(raw) ? raw : 'fft';
}

function cacheFile(pair: string, depth: string, lam: string): string {
	if (lam === 'fft') return join(CACHE_ROOT, `${pair}_d${depth}.json`);
	return join(CACHE_ROOT, lam, `${pair}_d${depth}.json`);
}

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	const lam = normalizeLam(url.searchParams.get('lam'));
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const file = cacheFile(pair, depth, lam);
	if (!existsSync(file)) return json({ cached: false });

	try {
		const payload = JSON.parse(readFileSync(file, 'utf-8'));
		if (!fingerprintMatches(Number(pair), payload.identity)) {
			console.warn(
				`[candidates] pair ${pair} identity mismatch: cached ${file} was written for different images (labels likely regenerated).`
			);
		}
		return json({ cached: true, ...payload });
	} catch {
		return json({ cached: false });
	}
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.pair_id !== 'number' || typeof body.depth !== 'number') {
		error(400, 'Expected { pair_id: number, depth: number, lam? }');
	}

	const { pair_id, depth } = body as { pair_id: number; depth: number; lam?: string };
	const lam = normalizeLam(body.lam);
	if (pair_id < 0 || pair_id >= pairCount()) {
		error(400, `Pair ${pair_id} does not exist (valid range 0..${pairCount() - 1})`);
	}
	const key = jobKey(pair_id, depth, 'candidates', lam);

	const existing = jobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	const state: JobState = {
		running: true,
		done: 0,
		total: 0,
		error: null,
		finishedAt: null,
		stage: null
	};
	jobs.set(key, state);

	const child = spawn(
		PYTHON,
		[SCRIPT, String(pair_id), '--cache-depth', String(depth), '--lam', lam],
		{ cwd: REPO_ROOT }
	);

	let stdout = '';
	let stderr = '';
	child.stdout.on('data', (chunk: Buffer) => {
		stdout += chunk.toString();
		const lines = stdout.split('\n');
		stdout = lines.pop() ?? '';
		for (const line of lines) {
			const trimmed = line.trim();
			if (!trimmed) continue;
			if (trimmed.startsWith('stage=') || trimmed.startsWith('done=')) {
				console.log(`[c2f candidates pair=${pair_id} d=${depth} lam=${lam}] ${trimmed}`);
			}
			const stage = trimmed.match(/stage=(\S+)/);
			if (stage) {
				state.stage = stage[1];
				const tot = trimmed.match(/\btotal=(\d+)/);
				if (tot) state.total = parseInt(tot[1], 10);
				const i = trimmed.match(/\bi=(\d+)/);
				if (i) state.done = Math.max(0, parseInt(i[1], 10) - 1);
				jobs.set(key, { ...state });
				continue;
			}
			const m = trimmed.match(/^done=(\d+)\s+total=(\d+)/);
			if (m) {
				state.done = parseInt(m[1], 10);
				state.total = parseInt(m[2], 10);
				state.stage = null;
				jobs.set(key, { ...state });
			}
		}
	});
	child.stderr.on('data', (chunk: Buffer) => {
		const text = chunk.toString();
		stderr += text;
		const errLines = text.split('\n').filter((l) => l.trim());
		for (const line of errLines) {
			console.error(`[c2f candidates pair=${pair_id} d=${depth} lam=${lam}] ${line}`);
		}
	});

	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		state.stage = null;
		if (code !== 0) {
			const lastLine = stderr.trim().split('\n').filter(Boolean).pop();
			state.error = lastLine || `Process exited with code ${code}`;
		}
		jobs.set(key, { ...state });
	});

	child.on('error', (err) => {
		state.running = false;
		state.error = err.message;
		state.finishedAt = Date.now();
		jobs.set(key, { ...state });
	});

	return json({ started: true, state });
};
