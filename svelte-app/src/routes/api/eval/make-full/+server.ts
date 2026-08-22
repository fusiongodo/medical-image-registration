import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { datasetFromUrl, normalizeDataset, pairCount, pairDir, type DatasetId } from '$lib/server/datasets';
import { layersReady } from '$lib/server/evalOverlay';
import { makeFullJobs, makeFullJobKey, type MakeFullJobState } from '$lib/eval/makeFullJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'regWSI', 'make_full.py');
const LAYERS = new Set(['he', 'ihc', 'ihc_warped']);

function parseLayers(raw: unknown): string[] {
	let list: string[] = [];
	if (typeof raw === 'string') {
		list = raw
			.split(',')
			.map((s) => s.trim())
			.filter(Boolean);
	} else if (Array.isArray(raw)) {
		list = raw.map((s) => String(s).trim()).filter(Boolean);
	}
	if (!list.length) list = ['he'];
	for (const layer of list) {
		if (!LAYERS.has(layer)) error(400, `invalid layer ${layer}`);
	}
	return list;
}

function mosaicsReady(dataset: DatasetId, pairId: number, layers: string[]): boolean {
	const dir = resolve(pairDir(dataset, pairId), 'full');
	const metaPath = resolve(dir, 'meta.json');
	if (!existsSync(metaPath)) return false;
	let nq = 2;
	try {
		nq = JSON.parse(readFileSync(metaPath, 'utf-8')).nq ?? 2;
	} catch {
		return false;
	}
	return layersReady(dir, nq, layers);
}

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const dataset = datasetFromUrl(url);
	const layers = parseLayers(url.searchParams.get('layers'));
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing pair');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount(dataset)) error(404, `pair ${pair} out of range`);
	const key = makeFullJobKey(dataset, pairId, layers);
	const job = makeFullJobs.get(key) ?? null;
	return json({
		ready: mosaicsReady(dataset, pairId, layers),
		job,
		layers,
		dataset
	});
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.pair_id !== 'number') {
		error(400, 'Expected { pair_id, dataset?, layers?, force? }');
	}
	const dataset = normalizeDataset(typeof body.dataset === 'string' ? body.dataset : 'muromi');
	const pairId = body.pair_id as number;
	const layers = parseLayers(body.layers);
	const force = body.force === true;
	if (pairId < 0 || pairId >= pairCount(dataset)) {
		error(400, `Pair ${pairId} out of range`);
	}

	const key = makeFullJobKey(dataset, pairId, layers);
	const existing = makeFullJobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing, layers });
	}

	if (!force && mosaicsReady(dataset, pairId, layers)) {
		const state: MakeFullJobState = {
			running: false,
			done: 1,
			total: 1,
			stage: 'cached',
			error: null,
			finishedAt: Date.now()
		};
		makeFullJobs.set(key, state);
		return json({ started: false, cached: true, state, layers });
	}

	const state: MakeFullJobState = {
		running: true,
		done: 0,
		total: layers.length,
		stage: 'start',
		error: null,
		finishedAt: null
	};
	makeFullJobs.set(key, state);

	const args = [SCRIPT, String(pairId), '--layers', ...layers];
	const child = spawn(PYTHON, args, {
		cwd: REPO_ROOT,
		env: { ...process.env, MVR_DATASET: dataset }
	});

	let stdout = '';
	let stderr = '';
	child.stdout.on('data', (chunk: Buffer) => {
		stdout += chunk.toString();
		for (const line of stdout.split('\n')) {
			const m = line.match(/done=(\d+)\s+total=(\d+)(?:\s+stage=(\S+))?/);
			if (m) {
				state.done = parseInt(m[1], 10);
				state.total = parseInt(m[2], 10);
				if (m[3]) state.stage = m[3];
			}
		}
	});
	child.stderr.on('data', (chunk: Buffer) => {
		stderr += chunk.toString();
	});
	child.on('close', (code) => {
		state.running = false;
		state.finishedAt = Date.now();
		if (code !== 0) {
			const lastLine = stderr.trim().split('\n').filter(Boolean).pop();
			state.error = lastLine || `Process exited with code ${code}`;
		} else {
			state.stage = 'done';
			state.done = state.total || layers.length;
		}
		makeFullJobs.set(key, { ...state });
	});

	return json({ started: true, state, layers });
};
