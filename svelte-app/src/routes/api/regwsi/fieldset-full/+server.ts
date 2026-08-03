import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { pairCount } from '$lib/server/pairs';
import { fieldsetJobs, fieldsetJobKey, type FieldsetJobState } from '$lib/regwsi/fieldsetJobs';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const SCRIPT = resolve(REPO_ROOT, 'regWSI', 'make_fieldset_full.py');
const ESTIMATORS = new Set(['tps', 'wendland']);

function layersReady(pairId: number, estimator: string): boolean {
	const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(pairId), 'full');
	const metaPath = resolve(dir, 'meta.json');
	if (!existsSync(metaPath)) return false;
	let nq = 2;
	try {
		nq = JSON.parse(readFileSync(metaPath, 'utf-8')).nq ?? 2;
	} catch {
		return false;
	}
	const layer = `ihc_fieldset_${estimator}`;
	for (let qy = 0; qy < nq; qy++) {
		for (let qx = 0; qx < nq; qx++) {
			if (!existsSync(resolve(dir, `${layer}_y${qy}_x${qx}.jpg`))) return false;
		}
	}
	return true;
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const estimator = url.searchParams.get('estimator') ?? 'tps';
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing pair');
	if (!ESTIMATORS.has(estimator)) error(400, 'estimator must be tps or wendland');
	const pairId = Number(pair);
	if (pairId < 0 || pairId >= pairCount()) error(404, `pair ${pair} out of range`);

	const key = fieldsetJobKey(pairId, estimator);
	const job = fieldsetJobs.get(key) ?? null;
	const ready = layersReady(pairId, estimator);
	let stamp: unknown = null;
	const stampPath = resolve(
		REPO_ROOT,
		'data',
		'regwsi',
		String(pairId),
		'full',
		`ihc_fieldset_${estimator}.stamp.json`
	);
	if (existsSync(stampPath)) {
		try {
			stamp = JSON.parse(readFileSync(stampPath, 'utf-8'));
		} catch {
			stamp = null;
		}
	}
	return json({ ready, job, stamp });
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json().catch(() => null);
	if (!body || typeof body.pair_id !== 'number') {
		error(400, 'Expected { pair_id: number, estimator?: tps|wendland, force?: boolean }');
	}
	const pairId = body.pair_id as number;
	const estimator = body.estimator === 'wendland' ? 'wendland' : 'tps';
	const force = body.force === true;
	if (pairId < 0 || pairId >= pairCount()) {
		error(400, `Pair ${pairId} out of range`);
	}

	const key = fieldsetJobKey(pairId, estimator);
	const existing = fieldsetJobs.get(key);
	if (existing?.running) {
		return json({ started: false, state: existing });
	}

	if (!force && layersReady(pairId, estimator)) {
		const state: FieldsetJobState = {
			running: false,
			done: 1,
			total: 1,
			stage: 'cached',
			error: null,
			finishedAt: Date.now()
		};
		fieldsetJobs.set(key, state);
		return json({ started: false, cached: true, state });
	}

	const state: FieldsetJobState = {
		running: true,
		done: 0,
		total: 4,
		stage: 'start',
		error: null,
		finishedAt: null
	};
	fieldsetJobs.set(key, state);

	const args = [SCRIPT, String(pairId), '--estimator', estimator];
	if (force) args.push('--force');
	const child = spawn(PYTHON, args, { cwd: REPO_ROOT });

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
			state.done = state.total || 4;
		}
		fieldsetJobs.set(key, { ...state });
	});

	return json({ started: true, state });
};
