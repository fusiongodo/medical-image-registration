import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');
export type DatasetId = 'muromi' | 'acrobat' | 'anhir';

const LABELS_PATH = resolve(REPO_ROOT, 'data', 'macos_labels.json');
const ACROBAT_PAIRS = resolve(REPO_ROOT, 'data', 'acrobat', 'pairs.json');
const ANHIR_PAIRS = resolve(REPO_ROOT, 'data', 'anhir', 'pairs.json');

export function normalizeDataset(raw: string | null | undefined): DatasetId {
	const v = (raw || 'muromi').trim().toLowerCase();
	if (v === 'acrobat' || v === 'acro') return 'acrobat';
	if (v === 'anhir') return 'anhir';
	return 'muromi';
}

export function usesPairTiffs(dataset: DatasetId): boolean {
	return dataset === 'acrobat' || dataset === 'anhir';
}

export function regwsiRoot(dataset: DatasetId): string {
	if (dataset === 'acrobat') return resolve(REPO_ROOT, 'data', 'acrobat', 'regwsi');
	if (dataset === 'anhir') return resolve(REPO_ROOT, 'data', 'anhir', 'regwsi');
	return resolve(REPO_ROOT, 'data', 'regwsi');
}

export function rigidRoot(dataset: DatasetId): string {
	if (dataset === 'acrobat') return resolve(REPO_ROOT, 'data', 'acrobat', 'rigid');
	if (dataset === 'anhir') return resolve(REPO_ROOT, 'data', 'anhir', 'rigid');
	return resolve(REPO_ROOT, 'data', 'rigid', 'light_v1');
}

export function rigidPath(dataset: DatasetId, pairId: number): string {
	return resolve(rigidRoot(dataset), `${pairId}.json`);
}

export function pairDir(dataset: DatasetId, pairId: number): string {
	return resolve(regwsiRoot(dataset), String(pairId));
}

export function datasetFromUrl(url: URL): DatasetId {
	return normalizeDataset(url.searchParams.get('dataset'));
}

function countPairsJson(path: string): number {
	if (!existsSync(path)) return 0;
	try {
		const j = JSON.parse(readFileSync(path, 'utf-8'));
		const pairs = Array.isArray(j) ? j : j.pairs;
		return Array.isArray(pairs) ? pairs.length : 0;
	} catch {
		return 0;
	}
}

export function pairCount(dataset: DatasetId = 'muromi'): number {
	if (dataset === 'acrobat') return countPairsJson(ACROBAT_PAIRS);
	if (dataset === 'anhir') return countPairsJson(ANHIR_PAIRS);
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
		{ id: 'acrobat', label: 'ACROBAT', pairCount: pairCount('acrobat') },
		{ id: 'anhir', label: 'ANHIR', pairCount: pairCount('anhir') }
	];
}
