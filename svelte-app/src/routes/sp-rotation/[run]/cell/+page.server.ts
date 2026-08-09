import type { PageServerLoad } from './$types';
import { error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');

export const load: PageServerLoad = ({ params, url }) => {
	const runId = params.run;
	const pair = Number(url.searchParams.get('pair'));
	const angle = Number(url.searchParams.get('angle'));
	if (!Number.isInteger(pair) || !Number.isFinite(angle)) {
		error(400, 'pair and angle required');
	}

	const manPath = resolve(REPO_ROOT, 'data', 'sp_rot_runs', runId, 'manifest.json');
	if (!existsSync(manPath)) error(404, 'run not found');
	const manifest = JSON.parse(readFileSync(manPath, 'utf-8'));

	const resultPath = resolve(
		REPO_ROOT,
		'data',
		'sp_rot_runs',
		runId,
		String(pair),
		String(angle),
		'result.json'
	);
	let result: Record<string, unknown> | null = null;
	if (existsSync(resultPath)) {
		try {
			result = JSON.parse(readFileSync(resultPath, 'utf-8'));
		} catch {
			result = null;
		}
	}

	const labelsPath = resolve(REPO_ROOT, 'data', 'sp_rot_runs', runId, 'labels.json');
	let label: string | null = null;
	let note: string | null = null;
	if (existsSync(labelsPath)) {
		try {
			const store = JSON.parse(readFileSync(labelsPath, 'utf-8'));
			const entry = store?.labels?.[`${pair}:${angle}`];
			if (entry) {
				label = entry.label ?? null;
				note = entry.note ?? null;
			}
		} catch {
			/* ignore */
		}
	}

	const angles: number[] = Array.isArray(manifest.angles) ? manifest.angles : [];
	const pairs: number[] = Array.isArray(manifest.pairs) ? manifest.pairs : [];
	const ai = angles.indexOf(angle);
	const pi = pairs.indexOf(pair);
	const nextAngle = ai >= 0 && ai < angles.length - 1 ? angles[ai + 1] : null;
	const prevAngle = ai > 0 ? angles[ai - 1] : null;
	const nextPair = pi >= 0 && pi < pairs.length - 1 ? pairs[pi + 1] : null;
	const prevPair = pi > 0 ? pairs[pi - 1] : null;

	return {
		runId,
		pair,
		angle,
		manifest,
		result,
		label,
		note,
		nav: { nextAngle, prevAngle, nextPair, prevPair, angles, pairs }
	};
};
