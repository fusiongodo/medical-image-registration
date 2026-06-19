import { json, error } from '@sveltejs/kit';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { RequestHandler } from './$types';
import type { ValidationStore } from '$lib/types';

const VALIDATION_PATH = resolve('..', 'data', 'quadtree_level_validation.json');

function read(): ValidationStore {
	if (!existsSync(VALIDATION_PATH)) return {};
	try {
		return JSON.parse(readFileSync(VALIDATION_PATH, 'utf-8'));
	} catch {
		return {};
	}
}

function write(store: ValidationStore) {
	writeFileSync(VALIDATION_PATH, JSON.stringify(store, null, 2), 'utf-8');
}

export const GET: RequestHandler = () => {
	return json(read());
};

export const POST: RequestHandler = async ({ request }) => {
	const body = await request.json();
	const { pair_id, depth, valid } = body;

	if (typeof pair_id !== 'number' || typeof depth !== 'number' || typeof valid !== 'boolean') {
		error(400, 'Expected { pair_id: number, depth: number, valid: boolean }');
	}

	const store = read();
	if (!store[String(pair_id)]) store[String(pair_id)] = {};
	store[String(pair_id)][String(depth)] = valid;
	write(store);

	return json({ ok: true });
};

export const DELETE: RequestHandler = async ({ request }) => {
	const body = await request.json();
	const { pair_id, depth } = body;

	if (typeof pair_id !== 'number') {
		error(400, 'Expected { pair_id: number, depth?: number }');
	}

	const store = read();
	if (typeof depth === 'number') {
		if (store[String(pair_id)]) {
			delete store[String(pair_id)][String(depth)];
		}
	} else {
		delete store[String(pair_id)];
	}
	write(store);

	return json({ ok: true });
};
