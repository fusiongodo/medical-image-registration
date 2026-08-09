import type { PageServerLoad } from './$types';
import { listDatasets, normalizeDataset, pairCount, regwsiRoot } from '$lib/server/datasets';
import { listSpRotRuns, matrixStatus, buildSummary } from '$lib/server/spRotStore';
import { existsSync } from 'fs';
import { resolve } from 'path';

export const load: PageServerLoad = ({ url }) => {
	const dataset = normalizeDataset(url.searchParams.get('dataset') || 'muromi');
	const preferredRun = (url.searchParams.get('run') || '').trim() || null;
	const n = pairCount(dataset);
	const root = regwsiRoot(dataset);
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(root, String(i));
		const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		pairs.push({ pairId: i, ready });
	}

	const allRuns = listSpRotRuns().runs as {
		id: string;
		dataset?: string;
		n_labels?: number;
	}[];
	const runs = allRuns.filter((r) => (r.dataset || 'muromi') === dataset);
	const initialRunId =
		(preferredRun && runs.some((r) => r.id === preferredRun) && preferredRun) ||
		runs.find((r) => (r.n_labels ?? 0) > 0)?.id ||
		runs[0]?.id ||
		null;

	let initialMatrix: ReturnType<typeof matrixStatus> | null = null;
	let initialSummary: ReturnType<typeof buildSummary> | null = null;
	if (initialRunId) {
		try {
			initialMatrix = matrixStatus(initialRunId);
		} catch {
			initialMatrix = null;
		}
		try {
			initialSummary = buildSummary(initialRunId);
		} catch {
			initialSummary = null;
		}
	}

	return {
		pairs,
		dataset,
		datasets: listDatasets(),
		preferredRun,
		runs,
		initialRunId,
		initialMatrix,
		initialSummary
	};
};
