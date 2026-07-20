import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { cropRequest } from '$lib/liveCropWorker';

// Whole-image greyscale preview (raw, no deskew warp) at grid*CNN resolution,
// used by the deskew landmarking page. `level` selects sharpness (2 -> 4x).
export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const level = url.searchParams.get('level');
	const side = url.searchParams.get('side');

	if (!pair || !level || !/^\d+$/.test(pair) || !/^\d+$/.test(level)) {
		error(400, 'Missing or invalid pair / level');
	}
	if (side !== 'he' && side !== 'ihc') {
		error(400, 'side must be he or ihc');
	}

	try {
		const payload = await cropRequest({
			op: 'whole',
			pair: Number(pair),
			level: Number(level),
			side
		});
		const buffer = Buffer.from(payload.png as string, 'base64');
		return new Response(buffer, {
			headers: { 'Content-Type': 'image/png', 'Cache-Control': 'no-store' }
		});
	} catch (err) {
		error(500, err instanceof Error ? err.message : 'live-crop worker failed');
	}
};
