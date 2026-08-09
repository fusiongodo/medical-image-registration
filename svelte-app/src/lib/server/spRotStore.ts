import { existsSync, readdirSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');
const SP_ROT_ROOT = resolve(REPO_ROOT, 'data', 'sp_rot_runs');

function readJson(path: string): Record<string, unknown> | null {
	if (!existsSync(path)) return null;
	try {
		const data = JSON.parse(readFileSync(path, 'utf-8'));
		return data && typeof data === 'object' ? (data as Record<string, unknown>) : null;
	} catch {
		return null;
	}
}

export function listSpRotRuns() {
	if (!existsSync(SP_ROT_ROOT)) return { runs: [] as unknown[] };
	const runs = [];
	for (const name of readdirSync(SP_ROT_ROOT)) {
		const dir = resolve(SP_ROT_ROOT, name);
		const man = readJson(resolve(dir, 'manifest.json'));
		if (!man) continue;
		const status = readJson(resolve(dir, 'status.json')) || {};
		const labelsRaw = readJson(resolve(dir, 'labels.json'));
		const labels =
			labelsRaw && typeof labelsRaw.labels === 'object' && labelsRaw.labels
				? (labelsRaw.labels as Record<string, { label?: string }>)
				: {};
		let n_pass = 0;
		let n_fail = 0;
		let n_unsure = 0;
		for (const v of Object.values(labels)) {
			const lab = String(v?.label || '').toLowerCase();
			if (lab === 'pass') n_pass += 1;
			else if (lab === 'fail') n_fail += 1;
			else if (lab === 'unsure') n_unsure += 1;
		}
		const n_labels = n_pass + n_fail + n_unsure;
		runs.push({
			id: name,
			name: (man.name as string) || name,
			dataset: (man.dataset as string) || 'muromi',
			pairs: (man.pairs as number[]) || [],
			angles: (man.angles as number[]) || [],
			created_at: man.created_at,
			status,
			n_labels,
			n_pass,
			n_fail,
			n_unsure
		});
	}
	runs.sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0));
	return { runs };
}

export function matrixStatus(runId: string) {
	const dir = resolve(SP_ROT_ROOT, runId);
	const man = readJson(resolve(dir, 'manifest.json'));
	if (!man) throw new Error(`no manifest for ${runId}`);
	const status = readJson(resolve(dir, 'status.json')) || {};
	const labelsRaw = readJson(resolve(dir, 'labels.json'));
	const labels =
		labelsRaw && typeof labelsRaw.labels === 'object' && labelsRaw.labels
			? (labelsRaw.labels as Record<string, { label?: string }>)
			: {};

	const pairs = ((man.pairs as number[]) || []).map(Number);
	const angles = ((man.angles as number[]) || []).map(Number);
	const cells = [];
	for (const pid of pairs) {
		for (const ang of angles) {
			const resultPath = resolve(dir, String(pid), String(ang), 'result.json');
			const res = readJson(resultPath);
			const key = `${pid}:${ang}`;
			const lab = labels[key]?.label ?? null;
			const entry: Record<string, unknown> = {
				pair_id: pid,
				angle: ang,
				state: 'missing',
				label: lab,
				n_inliers: null,
				rmse_px: null,
				rot_err_deg: null,
				trans_err_px: null,
				error: null
			};
			if (res) {
				if (res.error) {
					entry.state = 'error';
					entry.error = res.error;
				} else {
					entry.state = 'done';
					entry.n_inliers = res.n_inliers ?? null;
					const stats = (res.stats as Record<string, unknown>) || {};
					entry.rmse_px = stats.rmse_px ?? null;
					entry.rot_err_deg = res.rot_err_deg ?? null;
					entry.trans_err_px = res.trans_err_px ?? null;
				}
			}
			cells.push(entry);
		}
	}
	return { run_id: runId, manifest: man, status, cells };
}

const LABELS = ['pass', 'fail', 'unsure'] as const;

function mean(vals: number[]) {
	if (!vals.length) return null;
	return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function median(vals: number[]) {
	if (!vals.length) return null;
	const s = [...vals].sort((a, b) => a - b);
	const mid = Math.floor(s.length / 2);
	return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function metricStats(vals: number[]) {
	return { n: vals.length, mean: mean(vals), median: median(vals) };
}

function pushNum(arr: number[], val: unknown) {
	if (val == null) return;
	const f = Number(val);
	if (Number.isFinite(f)) arr.push(f);
}

export function buildSummary(runId: string) {
	const dir = resolve(SP_ROT_ROOT, runId);
	const man = readJson(resolve(dir, 'manifest.json'));
	if (!man) throw new Error(`no manifest for ${runId}`);
	const labelsRaw = readJson(resolve(dir, 'labels.json'));
	const labels =
		labelsRaw && typeof labelsRaw.labels === 'object' && labelsRaw.labels
			? (labelsRaw.labels as Record<string, { label?: string }>)
			: {};
	const pairs = ((man.pairs as number[]) || []).map(Number);
	const angles = ((man.angles as number[]) || []).map(Number);
	const by_angle: Record<string, unknown> = {};
	const metric_by_label: Record<string, Record<string, number[]>> = {};
	for (const lab of LABELS) {
		metric_by_label[lab] = {
			n_inliers: [],
			rmse_px: [],
			rot_err_deg: [],
			trans_err_px: []
		};
	}
	let cells_done = 0;
	let cells_failed = 0;

	for (const ang of angles) {
		const counts: Record<string, number> = {
			pass: 0,
			fail: 0,
			unsure: 0,
			unlabeled: 0
		};
		const metric_lists: Record<string, number[]> = {
			n_inliers: [],
			rmse_px: [],
			rot_err_deg: [],
			trans_err_px: []
		};
		for (const pid of pairs) {
			const res = readJson(resolve(dir, String(pid), String(ang), 'result.json'));
			const key = `${pid}:${ang}`;
			const lab = String(labels[key]?.label || '').toLowerCase() || null;
			if (lab && LABELS.includes(lab as (typeof LABELS)[number])) counts[lab] += 1;
			else counts.unlabeled += 1;

			if (!res) continue;
			if (res.error) {
				cells_failed += 1;
				continue;
			}
			cells_done += 1;
			const stats = (res.stats as Record<string, unknown>) || {};
			const vals: Record<string, unknown> = {
				n_inliers: res.n_inliers,
				rmse_px: stats.rmse_px,
				rot_err_deg: res.rot_err_deg,
				trans_err_px: res.trans_err_px
			};
			for (const name of Object.keys(metric_lists)) {
				pushNum(metric_lists[name], vals[name]);
				if (lab && LABELS.includes(lab as (typeof LABELS)[number])) {
					pushNum(metric_by_label[lab][name], vals[name]);
				}
			}
		}
		const labeled = counts.pass + counts.fail + counts.unsure;
		by_angle[String(ang)] = {
			counts,
			labeled,
			fail_rate: labeled ? counts.fail / labeled : null,
			metrics: Object.fromEntries(
				Object.entries(metric_lists).map(([k, v]) => [k, metricStats(v)])
			)
		};
	}

	return {
		run_id: runId,
		dataset: man.dataset,
		pairs,
		angles,
		cells_with_result: cells_done,
		cells_failed,
		by_angle,
		metrics_by_label: Object.fromEntries(
			LABELS.map((lab) => [
				lab,
				Object.fromEntries(
					Object.entries(metric_by_label[lab]).map(([k, v]) => [k, metricStats(v)])
				)
			])
		),
		built_at: Math.floor(Date.now() / 1000)
	};
}
