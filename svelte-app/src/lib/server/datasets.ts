import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');
export type DatasetId = 'muromi' | 'acrobat';

const LABELS_PATH = resolve(REPO_ROOT, 'data', 'macos_labels.json');
const ACROBAT_PAIRS = resolve(REPO_ROOT, 'data', 'acrobat', 'pairs.json');

export function normalizeDataset(raw: string | null | undefined): DatasetId {
	const v = (raw || 'muromi').trim().toLowerCase();
	if (v === 'acrobat' || v === 'acro') return 'acrobat';
	return 'muromi';
}

export function regwsiRoot(dataset: DatasetId): string {
	if (dataset === 'acrobat') return resolve(REPO_ROOT, 'data', 'acrobat', 'regwsi');
	return resolve(REPO_ROOT, 'data', 'regwsi');
}

export function pairDir(dataset: DatasetId, pairId: number): string {
	return resolve(regwsiRoot(dataset), String(pairId));
}

export function datasetFromUrl(url: URL): DatasetId {
	return normalizeDataset(url.searchParams.get('dataset'));
}

export function pairCount(dataset: DatasetId = 'muromi'): number {
	if (dataset === 'acrobat') {
		if (!existsSync(ACROBAT_PAIRS)) return 0;
		try {
			const j = JSON.parse(readFileSync(ACROBAT_PAIRS, 'utf-8'));
			const pairs = Array.isArray(j) ? j : j.pairs;
			return Array.isArray(pairs) ? pairs.length : 0;
		} catch {
			return 0;
		}
	}
	try {
		const json = JSON.parse(readFileSync(LABELS_PATH, 'utf-8'));
		return Array.isArray(json) && json.length > 0 ? json.length : 8;
	} catch {
		return 8;
	}
}

export function listDatasets(): { id: DatasetId; label: string; pairCount: number }[] {
	return [
		{ id: 'muromi', label: 'muROMI', pairCount: pairCount('muromi') },
		{ id: 'acrobat', label: 'ACROBAT', pairCount: pairCount('acrobat') }
	];
}
