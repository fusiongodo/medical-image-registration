import { error } from '@sveltejs/kit';
import { readFileSync, existsSync } from 'fs';
import { resolve, normalize } from 'path';
import type { RequestHandler } from './$types';

const ALLOWED_ROOTS = [
	normalize(resolve('..', 'data', 'cropped')),
	normalize(resolve('..', 'data', 'cropped_smooth')),
];

export const GET: RequestHandler = ({ url }) => {
	const rawPath = url.searchParams.get('path');
	if (!rawPath) error(400, 'Missing path');

	const absolute = normalize(resolve('..', rawPath));

	if (!ALLOWED_ROOTS.some((root) => absolute.startsWith(root))) {
		error(403, 'Forbidden');
	}

	if (!existsSync(absolute)) {
		error(404, 'Not found');
	}

	const buffer = readFileSync(absolute);
	return new Response(buffer, {
		headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=3600' }
	});
};
