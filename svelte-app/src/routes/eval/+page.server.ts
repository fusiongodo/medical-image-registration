import type { PageServerLoad } from './$types';
import { listDatasets, normalizeDataset, pairCount, regwsiRoot, rigidPath } from '$lib/server/datasets';
import { layersReady } from '$lib/server/evalOverlay';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');
const CANVAS_W = 16384;
const CANVAS_H = 11008;
const DF_INLIER_PX = 8;

type DatasetId = ReturnType<typeof normalizeDataset>;

function readJson(path: string): Record<string, unknown> | null {
	if (!existsSync(path)) return null;
	try {
		return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, unknown>;
	} catch {
		return null;
	}
}

function landmarkPoints(dir: string): { he: [number, number]; ihc: [number, number] }[] {
	const data = readJson(resolve(dir, 'landmarks.json'));
	const pts = data?.points;
	if (!Array.isArray(pts)) return [];
	return pts.filter(
		(p) =>
			Array.isArray(p?.he) &&
			p.he.length === 2 &&
			Array.isArray(p?.ihc) &&
			p.ihc.length === 2
	) as { he: [number, number]; ihc: [number, number] }[];
}

function canvasSize(dir: string): [number, number] {
	const full = readJson(resolve(dir, 'full', 'meta.json'));
	const meta = readJson(resolve(dir, 'meta.json'));
	const canvas = meta?.canvas;
	const w =
		typeof full?.w === 'number'
			? full.w
			: Array.isArray(canvas) && typeof canvas[0] === 'number'
				? canvas[0]
				: CANVAS_W;
	const h =
		typeof full?.h === 'number'
			? full.h
			: Array.isArray(canvas) && typeof canvas[1] === 'number'
				? canvas[1]
				: CANVAS_H;
	return [w, h];
}

function landmarkRigidInliers(
	dir: string,
	rigidRaw: unknown,
	dfW: number | null
): { nInliers: number; nTotal: number; inlierPx: number | null } | null {
	const points = landmarkPoints(dir);
	if (!points.length) return null;
	if (!Array.isArray(rigidRaw) || rigidRaw.length < 2) return null;
	const r0 = rigidRaw[0] as number[];
	const r1 = rigidRaw[1] as number[];
	if (!Array.isArray(r0) || !Array.isArray(r1) || r0.length < 3 || r1.length < 3) return null;
	const r00 = Number(r0[0]);
	const r01 = Number(r0[1]);
	const tx = Number(r0[2]);
	const r10 = Number(r1[0]);
	const r11 = Number(r1[1]);
	const ty = Number(r1[2]);
	if (![r00, r01, tx, r10, r11, ty].every((v) => Number.isFinite(v))) return null;
	const [w, h] = canvasSize(dir);
	const inlierPx = DF_INLIER_PX * (w / Math.max(dfW && dfW > 1 ? dfW : 3277, 1));
	let nInliers = 0;
	for (const p of points) {
		const px = r00 * p.ihc[0] + r01 * p.ihc[1] + tx;
		const py = r10 * p.ihc[0] + r11 * p.ihc[1] + ty;
		const dx = (px - p.he[0]) * w;
		const dy = (py - p.he[1]) * h;
		if (Math.hypot(dx, dy) <= inlierPx) nInliers += 1;
	}
	return { nInliers, nTotal: points.length, inlierPx };
}

function rigidInliers(
	dataset: DatasetId,
	pairId: number,
	dir: string
): { nInliers: number; nTotal: number; inlierPx: number | null } | null {
	const data = readJson(rigidPath(dataset, pairId));
	if (!data) return null;
	if (dataset !== 'muromi') {
		const stats = data.stats as { width?: number } | undefined;
		return landmarkRigidInliers(dir, data.rigid, typeof stats?.width === 'number' ? stats.width : null);
	}
	const nInliers = Number(data.n_inliers);
	const nTotal = Number(data.n_matches);
	if (!Number.isFinite(nInliers) || !Number.isFinite(nTotal) || nTotal <= 0) return null;
	const px = (data.hyperparams as { rigid_inlier_px?: number } | undefined)?.rigid_inlier_px;
	return {
		nInliers,
		nTotal,
		inlierPx: typeof px === 'number' && Number.isFinite(px) ? px : 3
	};
}

function mosaicReady(dir: string): boolean {
	const fullDir = resolve(dir, 'full');
	const metaPath = resolve(fullDir, 'meta.json');
	if (!existsSync(metaPath)) return false;
	try {
		const meta = JSON.parse(readFileSync(metaPath, 'utf-8'));
		const nq = typeof meta.nq === 'number' ? meta.nq : 2;
		return layersReady(fullDir, nq, ['he', 'ihc_warped']);
	} catch {
		return false;
	}
}

function mainSetMeta(pairId: number): { mainSetId: string | null; mainSetName: string | null } {
	let mainSetId: string | null = null;
	let mainSetName: string | null = null;
	const activePath = resolve(
		REPO_ROOT,
		'data',
		'curated_field_sets',
		'fft',
		'tps',
		String(pairId),
		'active.json'
	);
	if (!existsSync(activePath)) return { mainSetId, mainSetName };
	try {
		const active = JSON.parse(readFileSync(activePath, 'utf-8'));
		mainSetId = active.main_set_id ?? active.set_id ?? null;
		if (mainSetId) {
			const manifestPath = resolve(
				REPO_ROOT,
				'data',
				'curated_field_sets',
				'fft',
				'tps',
				String(pairId),
				mainSetId,
				'manifest.json'
			);
			if (existsSync(manifestPath)) {
				const m = JSON.parse(readFileSync(manifestPath, 'utf-8'));
				mainSetName = m.name ?? mainSetId;
			} else {
				mainSetName = mainSetId;
			}
		}
	} catch {
		mainSetId = null;
		mainSetName = null;
	}
	return { mainSetId, mainSetName };
}

export const load: PageServerLoad = ({ url }) => {
	const dataset = normalizeDataset(url.searchParams.get('dataset'));
	const n = pairCount(dataset);
	const root = regwsiRoot(dataset);
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(root, String(i));
		const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		const { mainSetId, mainSetName } = dataset === 'muromi' ? mainSetMeta(i) : { mainSetId: null, mainSetName: null };
		pairs.push({
			pairId: i,
			ready,
			mosaicReady: mosaicReady(dir),
			landmarkCount: landmarkPoints(dir).length,
			rigidInliers: rigidInliers(dataset, i, dir),
			mainSetId,
			mainSetName
		});
	}
	return { pairs, dataset, datasets: listDatasets() };
};
