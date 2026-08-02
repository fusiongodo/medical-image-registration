import type { PageServerLoad } from './$types';
import { pairCount } from '$lib/server/pairs';
import { existsSync, readFileSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');

function landmarkCount(dir: string): number {
	const path = resolve(dir, 'landmarks.json');
	if (!existsSync(path)) return 0;
	try {
		const data = JSON.parse(readFileSync(path, 'utf-8'));
		return Array.isArray(data.points) ? data.points.length : 0;
	} catch {
		return 0;
	}
}

export const load: PageServerLoad = () => {
	const n = pairCount();
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(i));
		const ready = existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		pairs.push({ pairId: i, ready, landmarkCount: landmarkCount(dir) });
	}
	return { pairs };
};
