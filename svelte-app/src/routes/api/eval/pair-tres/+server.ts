import { json, error } from '@sveltejs/kit';
import { existsSync, readdirSync, readFileSync } from 'fs';
import { join, resolve } from 'path';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const EVAL_ROOT = resolve(REPO_ROOT, 'data', 'eval_runs');
const LAMS = ['fft', 'superpoint_glue'] as const;

function readJson(path: string): Record<string, unknown> | null {
	try {
		return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
	} catch {
		return null;
	}
}

function treMean(path: string): number | null {
	const tre = readJson(path);
	const m = tre?.mean;
	return typeof m === 'number' && Number.isFinite(m) ? m : null;
}

function lamMean(batchDir: string, pair: number, lam: string, estimators: string[]): number | null {
	const order = ['wendland', ...estimators.filter((e) => e !== 'wendland')];
	const seen = new Set<string>();
	for (const est of order) {
		if (seen.has(est)) continue;
		seen.add(est);
		const v = treMean(join(batchDir, String(pair), lam, est, 'tre.json'));
		if (v != null) return v;
	}
	const lamDir = join(batchDir, String(pair), lam);
	if (!existsSync(lamDir)) return null;
	try {
		for (const name of readdirSync(lamDir)) {
			if (seen.has(name)) continue;
			const v = treMean(join(lamDir, name, 'tre.json'));
			if (v != null) return v;
		}
	} catch {
		return null;
	}
	return null;
}

export const GET: RequestHandler = ({ url }) => {
	const batch = url.searchParams.get('batch');
	if (!batch || !/^[\w.-]+$/.test(batch)) error(400, 'Missing or invalid batch');
	const batchDir = resolve(EVAL_ROOT, batch);
	if (!existsSync(batchDir)) error(404, `no batch ${batch}`);
	const man = readJson(join(batchDir, 'manifest.json'));
	if (!man) error(404, `no batch ${batch}`);
	const pairs = (Array.isArray(man.pairs) ? man.pairs : [])
		.map((p) => Number(p))
		.filter((p) => Number.isInteger(p));
	const estimators = (Array.isArray(man.estimators) ? man.estimators : []).map(String);
	const out: Record<string, { regwsi: number | null; fft: number | null; superpoint_glue: number | null }> =
		{};
	for (const p of pairs) {
		out[String(p)] = {
			regwsi: treMean(join(batchDir, String(p), 'regwsi', 'tre.json')),
			fft: lamMean(batchDir, p, LAMS[0], estimators),
			superpoint_glue: lamMean(batchDir, p, LAMS[1], estimators)
		};
	}
	return json({ batch, pairs: out });
};
