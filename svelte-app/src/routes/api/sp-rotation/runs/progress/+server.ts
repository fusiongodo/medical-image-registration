import { json, error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import { spRotJobs, spRotJobKey, type SpRotJobState } from '$lib/spRotJobs';

const REPO_ROOT = resolve('..');

function statusFromDisk(runId: string): SpRotJobState | null {
	const path = resolve(REPO_ROOT, 'data', 'sp_rot_runs', runId, 'status.json');
	if (!existsSync(path)) return null;
	try {
		const st = JSON.parse(readFileSync(path, 'utf-8')) as {
			state?: string;
			done?: number;
			total?: number;
			detail?: string;
			error?: string | null;
			failed?: number;
			skipped?: number;
			updated_at?: number;
		};
		const state = (st.state || 'idle').toLowerCase();
		const running = state === 'running' || state === 'cell' || state === 'gt';
		return {
			running,
			done: Number(st.done) || 0,
			total: Number(st.total) || 0,
			detail: st.detail || null,
			error: st.error ?? null,
			finishedAt: running ? null : (st.updated_at ?? null),
			stage: state,
			failed: Number(st.failed) || 0,
			skipped: Number(st.skipped) || 0
		};
	} catch {
		return null;
	}
}

export const GET: RequestHandler = ({ url }) => {
	const run = url.searchParams.get('run');
	if (!run) error(400, 'Missing run');
	const live = spRotJobs.get(spRotJobKey(run));
	if (live) return json(live);
	return json(statusFromDisk(run));
};
