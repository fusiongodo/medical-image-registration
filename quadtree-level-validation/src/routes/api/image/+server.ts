import { error } from '@sveltejs/kit';
import { readFileSync, existsSync } from 'fs';
import { resolve, normalize } from 'path';
import type { RequestHandler } from './$types';

const CROPPED_ROOT = resolve('..', 'data', 'cropped');

export const GET: RequestHandler = ({ url }) => {
	const rawPath = url.searchParams.get('path');
	if (!rawPath) error(400, 'Missing path');

	const absolute = resolve('..', rawPath);
	const normalizedRoot = normalize(CROPPED_ROOT);

	if (!absolute.startsWith(normalizedRoot)) {
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
