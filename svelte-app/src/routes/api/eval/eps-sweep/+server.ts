import { json } from '@sveltejs/kit';
import { existsSync, readdirSync, readFileSync } from 'fs';
import { join, resolve } from 'path';
import type { RequestHandler } from './$types';
import { datasetFromUrl } from '$lib/server/datasets';

const REPO_ROOT = resolve('..');
const EVAL_ROOT = resolve(REPO_ROOT, 'data', 'eval_runs');
const LAMS = ['fft', 'superpoint_glue'] as const;

type Lam = (typeof LAMS)[number];

function mean(vals: number[]): number | null {
	return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

function readJson(path: string): Record<string, unknown> | null {
	try {
		return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
	} catch {
		return null;
	}
}

function lamStats(batchId: string, pairs: number[], lam: Lam) {
	const means: number[] = [];
	const medians: number[] = [];
	for (const p of pairs) {
		const tre = readJson(join(EVAL_ROOT, batchId, String(p), lam, 'wendland', 'tre.json'));
		const m = tre?.mean;
		const d = tre?.median;
		if (typeof m === 'number' && Number.isFinite(m)) means.push(m);
		if (typeof d === 'number' && Number.isFinite(d)) medians.push(d);
	}
	return {
		n: means.length,
		mean_of_means: mean(means),
		mean_of_medians: mean(medians)
	};
}

export const GET: RequestHandler = ({ url }) => {
	const dataset = datasetFromUrl(url);
	if (!existsSync(EVAL_ROOT)) return json({ dataset, points: [] });

	type Cand = {
		eps: number;
		batch_id: string;
		created: number;
		pairs: number[];
		n_complete: number;
		status: Record<string, unknown> | null;
	};
	const byEps = new Map<number, Cand>();

	for (const name of readdirSync(EVAL_ROOT)) {
		if (name !== 'anhir-full' && !name.startsWith('anhir-wen')) continue;
		const man = readJson(join(EVAL_ROOT, name, 'manifest.json'));
		if (!man) continue;
		if (name !== 'anhir-full' && !name.startsWith('anhir-wen')) continue;
		const ds = typeof man.dataset === 'string' ? man.dataset : 'muromi';
		if (ds !== dataset) continue;
		const estimators = Array.isArray(man.estimators) ? man.estimators : [];
		if (!estimators.includes('wendland')) continue;
		const cfg = (man.config as Record<string, unknown> | undefined) || {};
		const eps = Number(cfg.wendland_eps);
		if (!Number.isFinite(eps)) continue;
		const pairs = (Array.isArray(man.pairs) ? man.pairs : [])
			.map((p) => Number(p))
			.filter((p) => Number.isInteger(p));
		let nComplete = 0;
		for (const p of pairs) {
			for (const lam of LAMS) {
				if (existsSync(join(EVAL_ROOT, name, String(p), lam, 'wendland', 'tre.json'))) {
					nComplete += 1;
				}
			}
		}
		const cand: Cand = {
			eps,
			batch_id: name,
			created: typeof man.created === 'number' ? man.created : 0,
			pairs,
			n_complete: nComplete,
			status: readJson(join(EVAL_ROOT, name, 'status.json'))
		};
		const prev = byEps.get(eps);
		const preferNew =
			!prev ||
			nComplete > prev.n_complete ||
			(nComplete === prev.n_complete && cand.created >= prev.created);
		if (preferNew) byEps.set(eps, cand);
	}

	const cands = [...byEps.values()].sort((a, b) => a.eps - b.eps);
	let shared = cands[0]?.pairs ?? [];
	if (cands.length > 1) {
		const sets = cands.map((c) => new Set(c.pairs));
		shared = cands[0].pairs.filter((p) => sets.every((s) => s.has(p)));
	}

	const points = cands.map((c) => {
		const pairs = shared.length ? shared : c.pairs;
		return {
			eps: c.eps,
			batch_id: c.batch_id,
			pairs,
			n_pairs: pairs.length,
			status: c.status,
			fft: lamStats(c.batch_id, pairs, 'fft'),
			superpoint_glue: lamStats(c.batch_id, pairs, 'superpoint_glue')
		};
	});

	return json({ dataset, points });
};
