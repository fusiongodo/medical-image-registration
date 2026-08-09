import { json, error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { batchJobs, batchJobKey, type BatchJobState } from '$lib/evalJobs';

const REPO_ROOT = resolve('..');

function statusFromDisk(batchId: string): BatchJobState | null {
	const path = resolve(REPO_ROOT, 'data', 'eval_runs', batchId, 'status.json');
	if (!existsSync(path)) return null;
	try {
		const st = JSON.parse(readFileSync(path, 'utf-8')) as {
			state?: string;
			done?: number;
			total?: number;
			detail?: string;
			error?: string | null;
			finished_at?: number;
		};
		const state = (st.state || 'idle').toLowerCase();
		const running = state === 'running' || state === 'ingest';
		return {
			running,
			done: Number(st.done) || 0,
			total: Number(st.total) || 0,
			detail: st.detail || null,
			error: st.error ?? null,
			finishedAt: st.finished_at ?? null,
			stage: state
		};
	} catch {
		return null;
	}
}

export const GET: RequestHandler = ({ url }) => {
	const batch = url.searchParams.get('batch');
	if (!batch) error(400, 'Missing batch');
	const live = batchJobs.get(batchJobKey(batch));
	if (live) return json(live);
	return json(statusFromDisk(batch));
};
