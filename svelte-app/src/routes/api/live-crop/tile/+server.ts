import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { cropRequest } from '$lib/liveCropWorker';

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const level = url.searchParams.get('level');
	const x = url.searchParams.get('x');
	const y = url.searchParams.get('y');
	const side = url.searchParams.get('side');
	const dx = url.searchParams.get('dx');
	const dy = url.searchParams.get('dy');

	if (
		!pair || !level || !x || !y ||
		!/^\d+$/.test(pair) || !/^\d+$/.test(level) || !/^\d+$/.test(x) || !/^\d+$/.test(y)
	) {
		error(400, 'Missing or invalid pair / level / x / y');
	}
	if (side !== 'he' && side !== 'ihc') {
		error(400, 'side must be he or ihc');
	}

	try {
		const payload = await cropRequest({
			op: 'crop',
			pair: Number(pair),
			level: Number(level),
			x: Number(x),
			y: Number(y),
			side,
			dx: dx ? Number(dx) : 0,
			dy: dy ? Number(dy) : 0
		});
		const b64 = payload.png as string;
		const buffer = Buffer.from(b64, 'base64');
		return new Response(buffer, {
			headers: { 'Content-Type': 'image/png', 'Cache-Control': 'no-store' }
		});
	} catch (err) {
		error(500, err instanceof Error ? err.message : 'live-crop worker failed');
	}
};
