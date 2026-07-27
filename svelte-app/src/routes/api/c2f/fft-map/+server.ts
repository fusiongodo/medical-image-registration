import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { cropRequest } from '$lib/liveCropWorker';

const NUM = /^-?\d+(\.\d+)?$/;

export const GET: RequestHandler = async ({ url }) => {
	const pair = url.searchParams.get('pair');
	const depth = url.searchParams.get('depth');
	const tile = url.searchParams.get('tile');
	const dx = url.searchParams.get('dx');
	const dy = url.searchParams.get('dy');
	const mx = url.searchParams.get('mx');
	const my = url.searchParams.get('my');

	if (!pair || !depth || !tile || !/^\d+$/.test(pair) || !/^\d+$/.test(depth)) {
		error(400, 'Missing or invalid pair / depth');
	}
	if (!/^\d+_\d+$/.test(tile)) {
		error(400, 'tile must be "x_y"');
	}
	for (const [name, v] of [['dx', dx], ['dy', dy], ['mx', mx], ['my', my]] as const) {
		if (v !== null && !NUM.test(v)) error(400, `invalid ${name}`);
	}

	try {
		const payload = await cropRequest({
			op: 'fft-map',
			pair: Number(pair),
			level: Number(depth),
			tile,
			dx: dx ? Number(dx) : 0,
			dy: dy ? Number(dy) : 0,
			mx: mx !== null ? Number(mx) : null,
			my: my !== null ? Number(my) : null
		});
		return json({
			image: `data:image/png;base64,${payload.png as string}`,
			w: payload.w,
			h: payload.h,
			cx: payload.cx,
			cy: payload.cy,
			peaks: payload.peaks,
			chosen: payload.chosen
		});
	} catch (err) {
		error(500, err instanceof Error ? err.message : 'fft-map worker failed');
	}
};
