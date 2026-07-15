import { json, error } from '@sveltejs/kit';
import { resolve, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import type { RequestHandler } from './$types';

const REPO_ROOT = resolve('..');
const SMOOTH_DIR = resolve(REPO_ROOT, 'data', 'smooth_c2f');

export const GET: RequestHandler = ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	if (!pair || !depth) error(400, 'Missing pair / depth');

	const file = join(SMOOTH_DIR, `${pair}_smooth_field.json`);
	if (!existsSync(file)) return json({});

	try {
		const data = JSON.parse(readFileSync(file, 'utf-8'));
		return json(data.depths?.[String(depth)] ?? {});
	} catch {
		return json({});
	}
};
