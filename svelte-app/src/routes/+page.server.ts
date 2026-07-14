import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import type { PageServerLoad } from './$types';
import type { ValidationStore } from '$lib/types';

const VALIDATION_PATH = resolve('..', 'data', 'quadtree_level_validation.json');

export const load: PageServerLoad = () => {
	let validation: ValidationStore = {};
	if (existsSync(VALIDATION_PATH)) {
		try {
			validation = JSON.parse(readFileSync(VALIDATION_PATH, 'utf-8'));
		} catch {
			validation = {};
		}
	}

	return { validation };
};
