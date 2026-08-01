import type { PageServerLoad } from './$types';
import { pairCount } from '$lib/server/pairs';
import { existsSync } from 'fs';
import { resolve } from 'path';

const REPO_ROOT = resolve('..');

export const load: PageServerLoad = () => {
	const n = pairCount();
	const pairs = [];
	for (let i = 0; i < n; i++) {
		const dir = resolve(REPO_ROOT, 'data', 'regwsi', String(i));
		const ready =
			existsSync(resolve(dir, 'preview', 'he.png')) &&
			existsSync(resolve(dir, 'preview', 'ihc_warped.png')) &&
			existsSync(resolve(dir, 'out', 'displacement_field.mha'));
		pairs.push({ pairId: i, ready });
	}
	return { pairs };
};
