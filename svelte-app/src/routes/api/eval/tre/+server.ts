import { json, error } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import {
	datasetFromUrl,
	normalizeDataset,
	pairCount,
	type DatasetId
} from '$lib/server/datasets';

const REPO_ROOT = resolve('..');
const PYTHON = resolve(REPO_ROOT, '.venv', 'bin', 'python3');
const BATCH_TRE = resolve(REPO_ROOT, 'setup', 'coarse_to_fine', 'eval_tre_cli.py');
const LEGACY_TRE = resolve(REPO_ROOT, 'regWSI', 'tre_cli.py');

function runJson(script: string, args: string[], dataset: DatasetId): Promise<unknown> {
	return new Promise((resolveP, rejectP) => {
		const child = spawn(PYTHON, [script, ...args], {
			cwd: REPO_ROOT,
			env: { ...process.env, MVR_DATASET: dataset }
		});
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
				rejectP(new Error(`bad tre output: ${(e as Error).message}`));
			}
		});
	});
}

function datasetForBatch(batchId: string): DatasetId | null {
	const path = resolve(REPO_ROOT, 'data', 'eval_runs', batchId, 'manifest.json');
	if (!existsSync(path)) return null;
	try {
		const man = JSON.parse(readFileSync(path, 'utf-8'));
		return normalizeDataset(man.dataset);
	} catch {
		return null;
	}
}

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const batch = url.searchParams.get('batch');
	if (!pair || !/^\d+$/.test(pair)) error(400, 'Missing or invalid pair');
	const pairId = Number(pair);
	const dataset = batch
		? (datasetForBatch(batch) ?? datasetFromUrl(url))
		: datasetFromUrl(url);
	if (pairId < 0 || pairId >= pairCount(dataset)) error(404, `pair ${pair} out of range`);
	try {
		if (batch) {
			const result = await runJson(BATCH_TRE, [String(pairId), '--batch', batch], dataset);
			return json(result);
		}
		const result = await runJson(LEGACY_TRE, [String(pairId)], dataset);
		return json(result);
	} catch (e) {
		error(500, `tre failed: ${(e as Error).message}`);
	}
};
